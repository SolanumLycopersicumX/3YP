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
    st.session_state.setdefault("records", [])
    st.session_state.setdefault("last_frame", None)


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


def get_source(
    mode: str,
    subject: int,
    start_epoch: int,
    stop_epoch: int,
    duration_sec: float,
):
    key = (mode, int(subject), int(start_epoch), int(stop_epoch), float(duration_sec))
    source = st.session_state.get("source")
    if st.session_state.get("source_key") == key and source is not None:
        return source

    old_source = st.session_state.get("source")
    stop = getattr(old_source, "stop", None)
    if callable(stop):
        stop()

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
    st.session_state["arm"] = ArmVisualizer()
    st.session_state["arm"].reset()
    return source


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
    st.session_state["records"].append(
        {
            "mode": frame.mode,
            "replay_index": frame.replay_index,
            "pred_name": frame.pred_name,
            "confidence": frame.confidence,
            "true_name": frame.true_name,
            "ctnet_action": frame.ctnet_predicted_action_name,
            "scripted_action": frame.scripted_demo_action_name,
            "executed_action": frame.executed_action_name,
            "executed_action_source": frame.executed_action_source,
            "trajectory_len": len(frame.trajectory_yz),
        }
    )


def render_dashboard(frame: DashboardFrame, max_channels: int) -> None:
    top_left, top_right = st.columns(2, gap="large")
    bottom_left, bottom_right = st.columns(2, gap="large")

    with top_left:
        st.subheader("Robotic Arm")
        if frame.arm_rgb is not None:
            st.image(frame.arm_rgb, channels="RGB", use_container_width=True)
        else:
            st.info("PyBullet image unavailable. The fallback trajectory is still running.")
        st.caption(f"Arm mode: {frame.status.get('arm_mode', 'unknown')}")
        if frame.replay_index is not None:
            if frame.replay_total is not None:
                st.caption(f"Replay epoch {frame.replay_index + 1} of {frame.replay_total}")
            else:
                st.caption(f"Replay step {frame.replay_index + 1}")

    with top_right:
        st.subheader("Raw vs Preprocessed EEG")
        st.pyplot(
            make_eeg_figure(
                frame.raw_eeg,
                frame.preprocessed_eeg_for_display,
                frame.sampling_rate,
                frame.channel_names,
                max_channels=max_channels,
            )
        )
        st.caption(
            "Visible preprocessing: 8-30 Hz bandpass. "
            f"Model input shape: {frame.model_input_shape}"
        )

    with bottom_left:
        st.subheader("Y-Z Trajectory")
        st.pyplot(make_trajectory_figure(frame.trajectory_yz))

    with bottom_right:
        st.subheader("Classification and Action")
        st.pyplot(make_probability_figure(frame.probabilities))
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


def _next_epoch(source, mode: str, has_frame: bool):
    if mode == "BrainFlow synthetic":
        return source.step()
    if not has_frame:
        return source.current()
    return source.step()


def _reset_run(source) -> None:
    reset = getattr(source, "reset", None)
    if callable(reset):
        reset()
    st.session_state["arm"] = ArmVisualizer()
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
        device = st.selectbox("Device", ["cpu", "cuda"], index=0)
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
        scripted_demo = st.checkbox("Scripted demo in synthetic mode", value=True)
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

    try:
        source = get_source(
            mode,
            int(subject),
            int(start_epoch),
            int(stop_epoch),
            float(duration_sec),
        )
        pipeline = get_pipeline(model_path, device)

        if reset_clicked:
            _reset_run(source)

        should_advance = step_clicked or run_enabled or st.session_state["last_frame"] is None
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

        if run_enabled:
            if playback_delay > 0:
                time.sleep(playback_delay)
            st.rerun()
    except Exception as exc:
        st.error(str(exc))
        if mode == "BrainFlow synthetic":
            st.info("BrainFlow synthetic mode requires BrainFlow and the OpenBCI stream wrapper.")
        elif mode == "Offline PhysioNet":
            st.info(
                "Offline PhysioNet mode requires MNE-compatible local data "
                "available to the PhysioNet loader."
            )


if __name__ == "__main__":
    main()
