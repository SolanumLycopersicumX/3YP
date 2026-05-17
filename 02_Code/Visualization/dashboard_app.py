from __future__ import annotations

import time
from pathlib import Path

import numpy as np

try:
    import streamlit as st
except ImportError as exc:
    raise SystemExit("streamlit is required. Install with: pip install streamlit") from exc

from arm_visualizer import ArmVisualizer
from data_sources import OfflinePhysioNetSource, SyntheticBrainFlowSource
from eeg_pipeline import DEFAULT_MODEL_PATH, EEGPipeline, preprocess_for_display
from models import DashboardFrame, build_action_decision, class_name, scripted_action_for_step
from plotting import (
    export_csv,
    export_jsonl,
    make_eeg_figure,
    make_probability_figure,
    make_trajectory_figure,
)


EXPORT_DIR = Path("03_Experiments/Visualization")
MODES = ["Offline PhysioNet", "BrainFlow synthetic"]


def init_state() -> None:
    st.session_state.setdefault("source", None)
    st.session_state.setdefault("source_key", None)
    st.session_state.setdefault("pipeline", None)
    st.session_state.setdefault("pipeline_key", None)
    st.session_state.setdefault("arm", ArmVisualizer())
    st.session_state.setdefault("arm_key", None)
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("last_frame", None)


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False
    try:
        probe = torch.ones((1,), device="cuda")
        _ = probe + 1
        if hasattr(torch.cuda, "synchronize"):
            torch.cuda.synchronize()
    except Exception:
        return False
    return True


def _device_options_and_default() -> tuple[list[str], int]:
    if _cuda_available():
        return ["cuda", "cpu"], 0
    return ["cpu", "cuda"], 0


def _resolve_runtime_device(requested_device: str) -> str:
    if str(requested_device).startswith("cuda") and not _cuda_available():
        return "cpu"
    return str(requested_device)


def _is_cuda_error(exc: Exception) -> bool:
    message = str(exc).lower()
    markers = (
        "cuda error",
        "cudnn",
        "no kernel image",
        "device-side assert",
        "invalid device function",
    )
    return any(marker in message for marker in markers)


def _source_hint_for_exception(mode: str, exc: Exception) -> str | None:
    if _is_cuda_error(exc):
        return None
    if mode == "BrainFlow synthetic":
        return "BrainFlow synthetic mode requires BrainFlow and the OpenBCI stream wrapper."
    if mode == "Offline PhysioNet":
        return (
            "Offline PhysioNet mode requires MNE-compatible local data "
            "available to the PhysioNet loader."
        )
    return None


def get_pipeline(model_path: str, device: str) -> EEGPipeline:
    key = (str(model_path), str(device))
    current = st.session_state.get("pipeline")
    if current is None or st.session_state.get("pipeline_key") != key:
        st.session_state["pipeline"] = EEGPipeline(
            model_path=Path(model_path),
            device=device,
        )
        st.session_state["pipeline_key"] = key
    return st.session_state["pipeline"]


def _close_arm_best_effort() -> None:
    close = getattr(st.session_state.get("arm"), "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def get_source(
    mode: str,
    subject: int,
    start_epoch: int,
    stop_epoch: int,
    duration_sec: float,
    arm_settings: dict | None = None,
):
    key = (mode, int(subject), int(start_epoch), int(stop_epoch), float(duration_sec))
    source = st.session_state.get("source")
    if st.session_state.get("source_key") == key and source is not None:
        return source

    old_source = st.session_state.get("source")
    stop = getattr(old_source, "stop", None)
    try:
        if callable(stop):
            stop()
    finally:
        _close_arm_best_effort()

    if mode == "Offline PhysioNet":
        source = OfflinePhysioNetSource(
            subject=int(subject),
            start_epoch=int(start_epoch),
            stop_epoch=int(stop_epoch),
        )
    elif mode == "BrainFlow synthetic":
        source = SyntheticBrainFlowSource(duration_sec=float(duration_sec))
    else:
        raise ValueError(f"Unknown source mode: {mode}")

    st.session_state["source"] = source
    st.session_state["source_key"] = key
    st.session_state["records"] = []
    st.session_state["last_frame"] = None
    st.session_state["arm"] = ArmVisualizer(**(arm_settings or {}))
    st.session_state["arm_key"] = _arm_config_key(arm_settings or {})
    st.session_state["arm"].reset()
    return source


def _arm_config_key(arm_settings: dict) -> tuple:
    target = tuple(arm_settings.get("camera_target", (0.0, 0.0, 0.25)))
    return (
        float(arm_settings.get("camera_yaw", 135.0)),
        float(arm_settings.get("camera_pitch", -25.0)),
        float(arm_settings.get("camera_distance", 0.85)),
        target,
    )


def _ensure_arm_visualizer(arm_settings: dict) -> None:
    key = _arm_config_key(arm_settings)
    if st.session_state.get("arm_key") == key and st.session_state.get("arm") is not None:
        return
    _close_arm_best_effort()
    st.session_state["arm"] = ArmVisualizer(**arm_settings)
    st.session_state["arm_key"] = key
    st.session_state["arm"].reset()
    st.session_state["records"] = []
    st.session_state["last_frame"] = None


def build_dashboard_frame(
    mode: str,
    source_epoch,
    pipeline: EEGPipeline,
    scripted_demo: bool,
) -> DashboardFrame:
    raw = np.asarray(source_epoch.raw_eeg)
    preprocessed = preprocess_for_display(raw, source_epoch.sampling_rate)
    prediction = pipeline.predict(preprocessed)

    scripted_action = None
    if mode == "BrainFlow synthetic" and scripted_demo:
        scripted_action = scripted_action_for_step(source_epoch.replay_index or 0)

    decision = build_action_decision(
        prediction.pred_class,
        scripted_demo_action=scripted_action,
    )
    arm_frame = st.session_state["arm"].step(decision.executed_action)

    status = dict(source_epoch.status)
    status.update(arm_frame.status)

    return DashboardFrame(
        mode=mode,
        raw_eeg=raw,
        preprocessed_eeg_for_display=preprocessed,
        model_input_shape=prediction.model_input_shape,
        sampling_rate=source_epoch.sampling_rate,
        channel_names=source_epoch.channel_names,
        pred_class=prediction.pred_class,
        pred_name=prediction.pred_name,
        probabilities=prediction.probabilities,
        confidence=prediction.confidence,
        true_label=source_epoch.true_label,
        true_name=class_name(source_epoch.true_label),
        ctnet_predicted_action=decision.ctnet_predicted_action,
        ctnet_predicted_action_name=decision.ctnet_predicted_action_name,
        scripted_demo_action=decision.scripted_demo_action,
        scripted_demo_action_name=decision.scripted_demo_action_name,
        executed_action=decision.executed_action,
        executed_action_name=decision.executed_action_name,
        executed_action_source=decision.executed_action_source,
        arm_rgb=arm_frame.arm_rgb,
        trajectory_yz=arm_frame.trajectory_yz,
        replay_index=source_epoch.replay_index,
        replay_total=source_epoch.replay_total,
        status=status,
    )


def record_frame(frame: DashboardFrame) -> None:
    position_y = None
    position_z = None
    if frame.trajectory_yz:
        position_y, position_z = frame.trajectory_yz[-1]

    probabilities = None
    if frame.probabilities is not None:
        probabilities = [float(value) for value in np.asarray(frame.probabilities).ravel()]

    st.session_state["records"].append(
        {
            "timestamp_unix": time.time(),
            "mode": frame.mode,
            "replay_index": frame.replay_index,
            "pred_name": frame.pred_name,
            "confidence": frame.confidence,
            "probabilities": probabilities,
            "true_name": frame.true_name,
            "ctnet_action": frame.ctnet_predicted_action_name,
            "scripted_action": frame.scripted_demo_action_name,
            "executed_action": frame.executed_action_name,
            "executed_action_source": frame.executed_action_source,
            "position_y": position_y,
            "position_z": position_z,
            "trajectory_len": len(frame.trajectory_yz),
        }
    )


def _format_source_status(status: dict) -> str:
    source_keys = ("mode", "source", "ground_truth")
    parts = [
        f"{key}={status[key]}"
        for key in source_keys
        if key in status and status[key] is not None
    ]
    return ", ".join(parts) if parts else "unavailable"


def render_dashboard(frame: DashboardFrame, max_channels: int) -> None:
    top_left, top_right = st.columns(2, gap="large")
    bottom_left, bottom_right = st.columns(2, gap="large")

    with top_left:
        st.subheader("Robotic Arm")
        if frame.arm_rgb is not None:
            st.image(frame.arm_rgb, channels="RGB", use_container_width=True)
        else:
            st.info("PyBullet image unavailable. The fallback trajectory is still running.")
        st.caption(
            "Arm mode: "
            f"{frame.status.get('arm_mode', 'unknown')} | "
            f"Model: {frame.status.get('arm_model', 'unknown')}"
        )
        if frame.replay_index is not None:
            if frame.replay_total is not None:
                st.caption(f"Replay epoch {frame.replay_index + 1} of {frame.replay_total}")
            else:
                st.caption(f"Replay step {frame.replay_index + 1}")

    with top_right:
        st.subheader("Raw vs Preprocessed EEG")
        with st.container(height=560, border=True):
            st.pyplot(
                make_eeg_figure(
                    frame.raw_eeg,
                    frame.preprocessed_eeg_for_display,
                    frame.sampling_rate,
                    frame.channel_names,
                    max_channels=max_channels,
                ),
                use_container_width=True,
            )
        st.caption(
            "Visible preprocessing: 8-30 Hz bandpass; amplitudes shown in uV. "
            f"Model input shape: {frame.model_input_shape}"
        )

    with bottom_left:
        st.subheader("Y-Z Trajectory")
        st.pyplot(make_trajectory_figure(frame.trajectory_yz))

    with bottom_right:
        st.subheader("Classification and Action")
        st.pyplot(make_probability_figure(frame.probabilities))
        with st.container(border=True):
            confidence = None if frame.confidence is None else f"{frame.confidence:.2f}"
            st.metric("Prediction", frame.pred_name or "none", confidence)

            if frame.true_name is not None:
                st.write(f"True label: `{frame.true_name}`")
            else:
                st.warning("Synthetic EEG has no ground-truth motor-imagery label.")

            st.write(f"CTNet action: `{frame.ctnet_predicted_action_name or 'none'}`")
            st.write(f"Scripted action: `{frame.scripted_demo_action_name or 'disabled'}`")
            st.write(
                f"Executed action: `{frame.executed_action_name or 'none'}` "
                f"from `{frame.executed_action_source}`"
            )
            st.caption(f"Data-source status: {_format_source_status(frame.status)}")


def _next_epoch(source, mode: str, has_frame: bool):
    if mode == "BrainFlow synthetic":
        return source.step()
    if not has_frame:
        return source.current()
    return source.step()


def _offline_replay_finished(mode: str, frame: DashboardFrame | None) -> bool:
    if mode != "Offline PhysioNet" or frame is None:
        return False
    if frame.replay_index is None or frame.replay_total is None:
        return False
    return frame.replay_index >= frame.replay_total - 1


def _should_advance(
    mode: str,
    frame: DashboardFrame | None,
    step_clicked: bool,
    run_enabled: bool,
) -> bool:
    if frame is None:
        return True
    if not (step_clicked or run_enabled):
        return False
    if _offline_replay_finished(mode, frame):
        return False
    return True


def _reset_run(source, arm_settings: dict | None = None) -> None:
    reset = getattr(source, "reset", None)
    if callable(reset):
        reset()
    _close_arm_best_effort()
    settings = arm_settings or {}
    st.session_state["arm"] = ArmVisualizer(**settings)
    st.session_state["arm_key"] = _arm_config_key(settings)
    st.session_state["arm"].reset()
    st.session_state["records"] = []
    st.session_state["last_frame"] = None


def _export_log() -> tuple[Path, Path]:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = EXPORT_DIR / "dashboard_run.jsonl"
    csv_path = EXPORT_DIR / "dashboard_run.csv"
    export_jsonl(json_path, st.session_state["records"])
    export_csv(csv_path, st.session_state["records"])
    return json_path, csv_path


def main() -> None:
    st.set_page_config(page_title="EEG Arm Observer", layout="wide")
    init_state()
    st.title("EEG Arm Observer Dashboard")

    with st.sidebar:
        mode = st.radio("Mode", MODES)
        model_path = st.text_input("Model path", value=str(DEFAULT_MODEL_PATH))
        device_options, device_index = _device_options_and_default()
        device = st.selectbox("Device", device_options, index=device_index)
        max_channels = st.selectbox("Displayed channels", [4, 8, 16, 32, 64], index=1)
        subject = st.number_input("Subject", min_value=1, max_value=109, value=1, step=1)
        start_epoch = st.number_input("Start epoch", min_value=0, value=0, step=1)
        stop_epoch = st.number_input("Stop epoch", min_value=0, value=3, step=1)
        duration_sec = st.number_input(
            "Synthetic duration (s)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.5,
        )
        scripted_demo = st.checkbox("Use scripted demo action in synthetic mode", value=True)
        playback_delay = st.slider(
            "Playback delay (s)",
            min_value=0.0,
            max_value=2.0,
            value=0.5,
            step=0.1,
        )
        run_enabled = st.checkbox("Run", value=False)
        step_clicked = st.button("Step")
        reset_clicked = st.button("Reset")
        export_clicked = st.button("Export log")
        with st.expander("Arm camera", expanded=False):
            camera_yaw = st.slider("Yaw", min_value=-180.0, max_value=180.0, value=135.0, step=5.0)
            camera_pitch = st.slider("Pitch", min_value=-89.0, max_value=10.0, value=-25.0, step=1.0)
            camera_distance = st.slider("Distance", min_value=0.35, max_value=2.5, value=0.85, step=0.05)
            target_x = st.number_input("Target X", value=0.0, step=0.05, format="%.2f")
            target_y = st.number_input("Target Y", value=0.0, step=0.05, format="%.2f")
            target_z = st.number_input("Target Z", value=0.25, step=0.05, format="%.2f")

    arm_settings = {
        "camera_yaw": float(camera_yaw),
        "camera_pitch": float(camera_pitch),
        "camera_distance": float(camera_distance),
        "camera_target": (float(target_x), float(target_y), float(target_z)),
    }

    try:
        source = get_source(
            mode,
            int(subject),
            int(start_epoch),
            int(stop_epoch),
            float(duration_sec),
            arm_settings=arm_settings,
        )
        _ensure_arm_visualizer(arm_settings)
        runtime_device = _resolve_runtime_device(device)
        if runtime_device != device:
            st.warning(
                "CUDA was selected but failed a runtime compatibility check. "
                "Running inference on CPU for this session."
            )
        pipeline = get_pipeline(model_path, runtime_device)

        if reset_clicked:
            _reset_run(source, arm_settings=arm_settings)

        should_advance = _should_advance(
            mode,
            st.session_state["last_frame"],
            step_clicked=step_clicked,
            run_enabled=run_enabled,
        )
        if should_advance:
            epoch = _next_epoch(
                source,
                mode,
                has_frame=st.session_state["last_frame"] is not None,
            )
            frame = build_dashboard_frame(
                mode,
                epoch,
                pipeline,
                scripted_demo=scripted_demo,
            )
            st.session_state["last_frame"] = frame
            record_frame(frame)

        frame = st.session_state["last_frame"]
        if frame is not None:
            render_dashboard(frame, int(max_channels))

        if export_clicked:
            json_path, csv_path = _export_log()
            st.success(f"Exported {json_path} and {csv_path}")

        replay_finished = _offline_replay_finished(mode, st.session_state["last_frame"])
        if replay_finished and (run_enabled or step_clicked):
            st.info("Offline replay reached the final selected epoch. Reset or change the range to replay again.")

        if run_enabled and not replay_finished:
            if playback_delay > 0:
                time.sleep(playback_delay)
            st.rerun()
    except Exception as exc:
        st.error(str(exc))
        if _is_cuda_error(exc):
            st.warning(
                "CUDA execution failed. Use CPU, or install a PyTorch/CUDA build "
                "compiled for this GPU architecture before selecting CUDA again."
            )
        else:
            hint = _source_hint_for_exception(mode, exc)
            if hint is not None:
                st.info(hint)


if __name__ == "__main__":
    main()
