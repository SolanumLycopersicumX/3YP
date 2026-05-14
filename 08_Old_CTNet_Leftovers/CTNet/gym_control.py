import argparse
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation

from arm_gym_env import SimpleArm2DEnv, ArmConfig
try:
    from pybullet_arm_env import PyBulletArmEnv, PBConfig
except Exception:
    PyBulletArmEnv = None
    PBConfig = None
try:
    from serial_arm_env import SerialArmEnv, SerialConfig
except Exception:
    SerialArmEnv = None
    SerialConfig = None


# Map class id to movement direction (same as control.py)
MOVE_MAP = {
    0: (-1.0, 0.0),  # left -> -Y
    1: (1.0, 0.0),   # right -> +Y
    2: (0.0, 1.0),   # up -> +Z
    3: (0.0, -1.0),  # down -> -Z
}


def number_class_channel(dataset_type: str) -> Tuple[int, int]:
    if dataset_type == "A":
        return 4, 22
    if dataset_type == "B":
        return 2, 3
    raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_data(dir_path: Path, dataset_type: str, subject_id: int, mode: str):
    suffix = "T" if mode == "train" else "E"
    mat_path = dir_path / f"{dataset_type}{subject_id:02d}{suffix}.mat"
    data_mat = loadmat(mat_path)
    return data_mat["data"], data_mat["label"]


def load_data_evaluate(dir_path: Path, dataset_type: str, subject_id: int, evaluate_mode: str):
    if evaluate_mode == "LOSO":
        x_train, y_train = None, None
        for i in range(1, 10):
            x1, y1 = load_data(dir_path, dataset_type, i, mode="train")
            x2, y2 = load_data(dir_path, dataset_type, i, mode="test")
            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
            if i == subject_id:
                x_test, y_test = x, y
            elif x_train is None:
                x_train, y_train = x, y
            else:
                x_train = np.concatenate([x_train, x], axis=0)
                y_train = np.concatenate([y_train, y], axis=0)
        return x_train, y_train, x_test, y_test
    else:
        x_train, y_train = load_data(dir_path, dataset_type, subject_id, mode="train")
        x_test, y_test = load_data(dir_path, dataset_type, subject_id, mode="test")
        return x_train, y_train, x_test, y_test


def expand_labels(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return labels
    if labels.min() == 1:
        labels = labels - 1
    return labels.astype(int)


def parse_args():
    p = argparse.ArgumentParser(description="Drive a Gymnasium 2D arm env using CTNet predictions.")
    p.add_argument("--subject", type=int, default=2)
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--evaluate-mode", choices=["LOSO", "subject-dependent"], default="subject-dependent")
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--step-size", type=float, default=0.03)
    p.add_argument("--render-mode", choices=["rgb_array", "human"], default="rgb_array")
    p.add_argument("--save-animation", type=Path, default=Path("outputs/gym_arm.gif"))
    p.add_argument("--save-figure", type=Path, default=Path("outputs/gym_summary.png"))
    p.add_argument("--demo-mode", choices=["pred", "ground_truth", "random"], default="pred",
                   help="pred: use model predictions; ground_truth: follow labels; random: random actions")
    p.add_argument("--sleep", type=float, default=0.0, help="Per-step delay (seconds) for human rendering.")
    p.add_argument("--block", action="store_true", help="Block at the end to keep the window open (human mode).")
    p.add_argument("--repeat", type=int, default=1, help="Repeat each action K times to slow down and lengthen the episode.")
    p.add_argument("--return-center", action="store_true", help="After each action, return to start pose (pulse mode).")
    p.add_argument("--pulse-hold", type=float, default=0.0, help="Hold time at displaced pose before returning (seconds).")
    p.add_argument("--center-sleep", type=float, default=0.0, help="Sleep after returning to center (seconds).")
    # Env choice and PyBullet options
    p.add_argument("--env", choices=["simple2d", "pybullet", "serial"], default="simple2d")
    p.add_argument("--urdf", type=str, default=None, help="Path to robot URDF (PyBullet). If omitted, uses KUKA iiwa.")
    p.add_argument("--pb-gui", action="store_true", help="Use PyBullet GUI in human render mode.")
    p.add_argument("--ee-link", type=int, default=None, help="End-effector link index (PyBullet). Auto if omitted.")
    p.add_argument("--pb-substeps", type=int, default=None, help="Physics substeps per control step (PyBullet). Default=24.")
    p.add_argument("--pb-timestep", type=float, default=None, help="Physics base timestep seconds (PyBullet). Default=1/240.")
    p.add_argument("--plane-urdf", type=str, default=None, help="自定义地面 URDF 路径（如 plane.urdf）。")
    p.add_argument("--no-plane", action="store_true", help="不加载地面（禁用 plane）。")
    # Optional grid floor (debug lines)
    p.add_argument("--grid-floor", action="store_true", help="绘制网格地面（调试线）。")
    p.add_argument("--grid-size", type=float, default=1.5, help="网格半尺寸（米）。")
    p.add_argument("--grid-step", type=float, default=0.1, help="网格线间距（米）。")
    p.add_argument("--grid-height", type=float, default=0.0, help="网格所在高度 z（米）。")
    # Serial (real robot) options
    p.add_argument("--serial-port", type=str, default=None, help="串口端口，例如 /dev/ttyUSB0")
    p.add_argument("--serial-baud", type=int, default=1_000_000, help="串口波特率，默认 1Mbps")
    p.add_argument("--serial-timeout", type=float, default=0.02, help="串口超时，秒")
    p.add_argument("--serial-move-time", type=int, default=250, help="目标到达时间(ms)，位置模式下更平滑")
    p.add_argument("--y-min", type=float, default=None, help="Y lower bound of workspace.")
    p.add_argument("--y-max", type=float, default=None, help="Y upper bound of workspace.")
    p.add_argument("--z-min", type=float, default=None, help="Z lower bound of workspace.")
    p.add_argument("--z-max", type=float, default=None, help="Z upper bound of workspace.")
    p.add_argument("--start-y", type=float, default=None, help="Initial Y target (overrides auto).")
    p.add_argument("--start-z", type=float, default=None, help="Initial Z target (overrides auto).")
    p.add_argument("--pb-home-json", type=Path, default=None, help="JSON with initial joint positions (from pb_joint_tuner save)")
    p.add_argument("--pb-home-mid", action="store_true", help="Initialize all movable joints to mid (min+max)/2")
    p.add_argument("--block-pb", action="store_true", help="PyBullet专用：动作结束后按回车再关闭窗口（human+--pb-gui 有效）")
    # TUI monitor similar to joint tuner
    p.add_argument("--tui", action="store_true", help="启用终端界面监视（顶部指令+状态，底部键提示）")
    p.add_argument("--tui-interval", type=float, default=0.2, help="TUI 刷新间隔（秒）")
    # Prediction outputs
    p.add_argument("--show-probs", action="store_true", help="显示模型输出的类别概率（pred 模式有效）")
    p.add_argument("--save-preds", type=Path, default=None, help="保存每步预测与概率到CSV")
    # Joint-drive mode options (map directions to joint pos instead of EE Y/Z)
    p.add_argument("--joint-drive", action="store_true", help="使用关节角驱动：左右→shoulder_pan，上下→wrist_flex（可配置）")
    p.add_argument("--joint-step", type=float, default=0.05, help="关节每步增量（弧度），默认0.05")
    p.add_argument("--joint-lr-name", type=str, default="shoulder_pan", help="左右映射的关节名")
    p.add_argument("--joint-ud-name", type=str, default="elbow_flex", help="上下映射的关节名")
    p.add_argument("--invert-lr", action="store_true", help="反转左右方向")
    p.add_argument("--invert-ud", action="store_true", help="反转上下方向")
    return p.parse_args()


def load_model(dataset_type: str, subject_id: int, device: torch.device, model_path: Path | None):
    # Delay heavy imports; CTNet_model pulls in torch and others
    import sys
    from CTNet_model import (
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    )
    try:
        from torch.serialization import add_safe_globals, safe_globals  # torch>=2.6
    except Exception:
        add_safe_globals = None
        safe_globals = None

    if model_path is None:
        model_path = Path("./models/new") / f"model_{subject_id}.pth"

    # Robust checkpoint loader for PyTorch>=2.6 (weights_only default True)
    safe_types = [
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    ]

    def _alias_main():
        # Provide legacy names under __main__ for pickled checkpoints
        main_mod = sys.modules.setdefault("__main__", sys.modules.get("__main__"))
        for cls in safe_types:
            setattr(main_mod, cls.__name__, cls)

    def _extract_state(state):
        # Normalize different checkpoint formats to a plain state_dict
        if hasattr(state, "state_dict") and callable(getattr(state, "state_dict")):
            return state.state_dict()
        if isinstance(state, dict):
            # common keys
            for k in ["state_dict", "model_state", "model", "net", "weights"]:
                if k in state and isinstance(state[k], (dict,)):
                    # sometimes nested one more level
                    inner = state[k]
                    if hasattr(inner, "state_dict"):
                        return inner.state_dict()
                    if isinstance(inner, dict):
                        # Heuristic: looks like a state dict if most values are tensors
                        return inner
            return state
        return state

    def _load_ckpt(path: Path):
        # Strategy 1: safe weights-only path allowing all relevant classes
        if safe_globals is not None:
            try:
                _alias_main()
                with safe_globals(safe_types):
                    obj = torch.load(path, map_location=device, weights_only=True)
                return _extract_state(obj)
            except Exception:
                pass

        # Strategy 2: full unpickle with alias to real class
        try:
            _alias_main()
            return torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=device)
        except Exception as e1:
            # Strategy 3: allowlist real class and try default loader
            if add_safe_globals is not None:
                try:
                    add_safe_globals(safe_types)
                    _alias_main()
                    return torch.load(path, map_location=device)
                except Exception:
                    pass
            raise e1

    state = _load_ckpt(model_path)
    state = _extract_state(state)

    _, ch = number_class_channel(dataset_type)
    model = EEGTransformer(
        heads=2,
        emb_size=16,
        depth=6,
        database_type=dataset_type,
        eeg1_f1=8,
        eeg1_D=2,
        eeg1_kernel_size=64,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.5,
        eeg1_number_channel=ch,
        flatten_eeg1=240,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_actions(model, device, test_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    all_probs = []
    with torch.no_grad():
        for i in range(test_tensor.shape[0]):
            x = test_tensor[i].unsqueeze(0).to(device)
            logits = model(x)
            if isinstance(logits, tuple):
                _, logits = logits
            prob = F.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
            cls = int(np.argmax(prob))
            preds.append(cls)
            all_probs.append(prob)
    return np.array(preds, dtype=int), np.array(all_probs)


def to_frames_animation(frames: List[np.ndarray], save_path: Path, fps: int = 2):
    # Create an animation from raw frames using matplotlib + PillowWriter
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(frames[0])
    ax.axis("off")

    def update(i):
        im.set_data(frames[i])
        return [im]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=int(1000 / max(1, fps)), blit=True)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def plot_summary(path_y: List[float], path_z: List[float], actions: List[int], save_path: Path):
    # Count directions
    names = ["left", "right", "up", "down"]
    counts = [sum(1 for a in actions if a == i) for i in range(4)]
    dy = np.diff(np.array(path_y))
    dz = np.diff(np.array(path_z))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # 1) Cumulative path
    axs[0].plot(path_y, path_z, "-o")
    axs[0].set_title("End-effector Path (YZ)")
    axs[0].set_xlabel("Y")
    axs[0].set_ylabel("Z")
    axs[0].grid(alpha=0.3)

    # 2) Per-step vectors (quiver)
    origins_y = np.array(path_y[:-1])
    origins_z = np.array(path_z[:-1])
    axs[1].quiver(origins_y, origins_z, dy, dz, angles='xy', scale_units='xy', scale=1, width=0.003)
    axs[1].set_title("Per-step Displacements")
    axs[1].set_xlabel("Y")
    axs[1].set_ylabel("Z")
    axs[1].grid(alpha=0.3)

    # 3) Direction histogram
    axs[2].bar(names, counts, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
    axs[2].set_title("Direction Counts")
    axs[2].set_ylabel("Steps")
    for i, c in enumerate(counts):
        axs[2].text(i, c + 0.1, str(c), ha='center', va='bottom')

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    # Configure threading to reduce OpenMP contention
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    os.environ.setdefault("KMP_AFFINITY", "none")
    os.environ.setdefault("KMP_BLOCKTIME", "0")
    os.environ.setdefault("KMP_SETTINGS", "0")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_CREATE_SHM", "0")

    device = torch.device(args.device)

    # Load data
    _, _, test_data, test_labels = load_data_evaluate(args.data_dir, args.dataset, args.subject, args.evaluate_mode)
    test_data = np.expand_dims(test_data, axis=1)  # [N,1,22,1000]
    mean, std = np.mean(test_data), np.std(test_data) or 1.0
    test_data = (test_data - mean) / std
    test_tensor = torch.tensor(test_data, dtype=torch.float32, device=device)
    labels = expand_labels(test_labels)

    # Select actions source
    actions = None
    probs_matrix = None
    if args.demo_mode == "ground_truth":
        actions = labels.copy()
    elif args.demo_mode == "random":
        rng = np.random.default_rng(2024)
        actions = rng.integers(low=0, high=4, size=min(args.num_trials, len(test_tensor)))
    else:
        model = load_model(args.dataset, args.subject, device, args.model_path)
        preds, probs = predict_actions(model, device, test_tensor)
        actions = preds
        probs_matrix = probs

    # Truncate to requested number of trials
    actions = np.asarray(actions, dtype=int)[: args.num_trials]
    # Optionally repeat each action to slow down / lengthen
    if args.repeat and args.repeat > 1:
        actions = np.repeat(actions, args.repeat)

    # Create env
    if args.env == "pybullet":
        if PyBulletArmEnv is None:
            raise RuntimeError("PyBullet environment requested but pybullet_arm_env is unavailable.")
        substeps = args.pb_substeps if args.pb_substeps is not None else 24
        timestep = args.pb_timestep if args.pb_timestep is not None else (1.0 / 240.0)
        y_limits = (-0.5, 0.5)
        z_limits = (0.0, 1.0)
        if args.y_min is not None and args.y_max is not None:
            y_limits = (float(args.y_min), float(args.y_max))
        if args.z_min is not None and args.z_max is not None:
            z_limits = (float(args.z_min), float(args.z_max))
        # Load initial joint positions if provided
        home_positions = None
        if args.pb_home_json is not None:
            try:
                import json
                with open(args.pb_home_json, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Accept list of objects {index, position} or dict {index: position}
                if isinstance(data, dict):
                    home_positions = {int(k): float(v) for k, v in data.items() if k is not None}
                elif isinstance(data, list):
                    home_positions = {int(item['index']): float(item['position']) for item in data if 'index' in item and 'position' in item}
            except Exception as e:
                print(f"Warning: failed to load pb_home_json: {e}")

        pb_cfg = PBConfig(
            step_size=args.step_size,
            max_steps=len(actions),
            substeps=substeps,
            time_step=timestep,
            use_gui=args.pb_gui,
            urdf_path=args.urdf,
            ee_link_index=args.ee_link,
            y_limits=y_limits,
            z_limits=z_limits,
            start_y=args.start_y,
            start_z=args.start_z,
            home_mid=bool(args.pb_home_mid),
            home_joint_positions=home_positions,
            joint_drive=bool(args.joint_drive),
            joint_lr_name=args.joint_lr_name,
            joint_ud_name=args.joint_ud_name,
            joint_step=float(args.joint_step),
            invert_lr=bool(args.invert_lr),
            invert_ud=bool(args.invert_ud),
            load_plane=not bool(args.no_plane),
            plane_urdf=args.plane_urdf,
            grid_floor=bool(args.grid_floor),
            grid_size=float(args.grid_size),
            grid_step=float(args.grid_step),
            grid_height=float(args.grid_height),
        )
        env = PyBulletArmEnv(pb_cfg, render_mode=args.render_mode)
    elif args.env == "serial":
        if SerialArmEnv is None:
            raise RuntimeError("Serial 环境不可用（导入失败）")
        if not args.joint_drive:
            raise RuntimeError("serial 仅支持 --joint-drive（基于关节增量控制）")
        if args.serial_port is None:
            raise RuntimeError("请指定 --serial-port，例如 /dev/ttyUSB0")
        # 将 joint_step 视为弧度（与 PyBullet 一致），在 SerialEnv 内部换算为 ticks
        scfg = SerialConfig(
            port=args.serial_port,
            baud=int(args.serial_baud),
            timeout=float(args.serial_timeout),
            joint_lr_name=args.joint_lr_name,
            joint_ud_name=args.joint_ud_name,
            joint_step_rad=float(args.joint_step),
            invert_lr=bool(args.invert_lr),
            invert_ud=bool(args.invert_ud),
            max_steps=len(actions),
            move_time_ms=int(args.serial_move_time) if args.serial_move_time is not None else None,
        )
        env = SerialArmEnv(scfg, render_mode=args.render_mode)
    else:
        y_limits = (-1.0, 1.0)
        z_limits = (-1.0, 1.0)
        if args.y_min is not None and args.y_max is not None:
            y_limits = (float(args.y_min), float(args.y_max))
        if args.z_min is not None and args.z_max is not None:
            z_limits = (float(args.z_min), float(args.z_max))
        cfg = ArmConfig(step_size=args.step_size, max_steps=len(actions), y_limits=y_limits, z_limits=z_limits, start_y=args.start_y, start_z=args.start_z)
        env = SimpleArm2DEnv(cfg, render_mode=args.render_mode)
    obs, info = env.reset()

    frames: List[np.ndarray] = []
    path_y = [float(obs[0])]
    path_z = [float(obs[1])]
    taken_actions: List[int] = []
    pred_rows = []
    # label names per dataset
    n_class, _ = number_class_channel(args.dataset)
    if n_class == 4:
        label_names = ["left", "right", "up", "down"]
    elif n_class == 2:
        label_names = ["left", "right"]
    else:
        label_names = [f"c{i}" for i in range(n_class)]

    import time

    def run_step(a: int):
        nonlocal obs
        o, reward, terminated, truncated, _ = env.step(int(a))
        taken_actions.append(int(a))
        path_y.append(float(o[0]))
        path_z.append(float(o[1]))
        # record prediction row if available
        step_idx = len(taken_actions) - 1
        row = {"step": step_idx, "action": int(a), "y": float(o[0]), "z": float(o[1])}
        if probs_matrix is not None and step_idx < len(probs_matrix):
            for ci in range(probs_matrix.shape[1]):
                row[f"prob_{label_names[ci] if ci < len(label_names) else ci}"] = float(probs_matrix[step_idx, ci])
        pred_rows.append(row)
        if args.show_probs and not args.tui and probs_matrix is not None and step_idx < len(probs_matrix):
            parts = []
            for i in range(len(label_names)):
                p = float(probs_matrix[step_idx, i]) if i < probs_matrix.shape[1] else 0.0
                parts.append(f"{label_names[i]}={p:.2f}")
            print(f"[step {step_idx}] action={a} | " + "  ".join(parts))
        if args.render_mode == "rgb_array":
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        elif args.render_mode == "human":
            env.render()
            if args.sleep > 0:
                time.sleep(args.sleep)
        # Hold
        if args.pulse_hold > 0 and args.render_mode == "human":
            t_end = time.time() + args.pulse_hold
            while time.time() < t_end:
                if args.tui:
                    draw_tui(status_only=True)
                time.sleep(min(args.tui_interval, 0.05))
        # Return center
        if args.return_center:
            center_fn = getattr(env, "center", None)
            if callable(center_fn):
                center_fn()
            if args.render_mode == "human" and args.center_sleep > 0:
                t_end = time.time() + args.center_sleep
                while time.time() < t_end:
                    if args.tui:
                        draw_tui(status_only=True)
                    time.sleep(min(args.tui_interval, 0.05))
        return o, reward, terminated, truncated

    # TUI support
    def draw_tui(status_only: bool = False):
        try:
            import curses
        except Exception:
            return
        # no-op; real drawing handled in curses loop
        return

    if args.tui:
        try:
            import curses
            import locale
            locale.setlocale(locale.LC_ALL, '')

            def tui(stdscr):
                curses.curs_set(0)
                stdscr.nodelay(True)
                stdscr.timeout(int(max(0.05, args.tui_interval) * 1000))

                def get_status_text():
                    # Action name
                    act_names = {i: (label_names[i] if i < len(label_names) else str(i)) for i in range(n_class)}
                    curr_idx = len(taken_actions)
                    total = len(actions)
                    curr_act = act_names.get(actions[curr_idx] if curr_idx < total else -1, "-")
                    # Env status
                    try:
                        st = env.get_status()
                    except Exception:
                        st = {"ee_xyz": (float('nan'), float('nan'), float('nan')), "target_yz": (float('nan'), float('nan')), "step": curr_idx, "max_steps": len(actions)}
                    ee = st.get("ee_xyz", (np.nan, np.nan, np.nan))
                    yz = st.get("target_yz", (np.nan, np.nan))
                    # current probs
                    curr_probs = None
                    if probs_matrix is not None and curr_idx < len(probs_matrix):
                        curr_probs = probs_matrix[curr_idx]
                    return curr_idx, total, curr_act, ee, yz, curr_probs

                def draw_all(paused: bool):
                    stdscr.erase()
                    h, w = stdscr.getmaxyx()
                    avail_w = max(0, w - 2)
                    # header commands (two columns)
                    pairs = [
                        ("暂停/继续(空格)", "space (pause/resume)"),
                        ("单步(n)", "n (step once)"),
                        ("退出(q)", "q (quit)")
                    ]
                    row = 0
                    col_w = max(20, avail_w // 2)
                    for cn, en in pairs:
                        if row >= h - 4:
                            break
                        left = cn[: max(0, col_w - 1)]
                        right = en[: max(0, avail_w - col_w - 1)]
                        try:
                            stdscr.addstr(row, 1, left)
                            if col_w < avail_w:
                                stdscr.addstr(row, 1 + col_w, right)
                        except Exception:
                            pass
                        row += 1
                    # status
                    idx, total, act, ee, yz, curr_probs = get_status_text()
                    lines = [
                        f"状态: {'暂停' if paused else '运行'} | 动作 {idx}/{total} -> {act}",
                        f"EE: x={ee[0]:.4f}, y={ee[1]:.4f}, z={ee[2]:.4f}",
                        f"目标: Y={yz[0]:.4f}, Z={yz[1]:.4f}",
                    ]
                    for ln in lines:
                        if row >= h - 2:
                            break
                        try:
                            stdscr.addstr(row, 1, ln[: max(0, avail_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    # probabilities line
                    if args.show_probs and curr_probs is not None and row < h - 2:
                        parts = []
                        for i in range(len(label_names)):
                            p = float(curr_probs[i]) if i < len(curr_probs) else 0.0
                            parts.append(f"{label_names[i]}={p:.2f}")
                        prob_line = "Probs: " + "  ".join(parts)
                        try:
                            stdscr.addstr(row, 1, prob_line[: max(0, avail_w - 1)])
                        except Exception:
                            pass
                        row += 1
                    stdscr.box()
                    stdscr.refresh()

                paused = False
                i = 0
                draw_all(paused)
                while True:
                    ch = stdscr.getch()
                    if ch in (ord('q'), ord('Q')):
                        break
                    if ch in (ord(' '),):
                        paused = not paused
                        draw_all(paused)
                        continue
                    if ch in (ord('n'), ord('N')) and i < len(actions):
                        run_step(actions[i])
                        i += 1
                        draw_all(paused)
                        continue
                    if not paused and i < len(actions):
                        run_step(actions[i])
                        i += 1
                        draw_all(paused)
                        continue
                    if i >= len(actions):
                        # finished, keep UI until 'q'
                        draw_all(paused)
                        time.sleep(max(0.05, args.tui_interval))
                        continue

            import curses
            curses.wrapper(tui)
        except Exception as e:
            print(f"TUI 出错：{e}")
    else:
        # non-TUI original loop
        for t, a in enumerate(actions):
            obs, reward, terminated, truncated = run_step(a)
            if terminated or truncated:
                break

    # Keep window open if requested in human mode (Matplotlib)
    if args.render_mode == "human" and args.block:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass
    # PyBullet-specific blocking: wait for Enter before closing GUI
    if args.env == "pybullet" and args.render_mode == "human" and args.pb_gui and args.block_pb:
        try:
            input("按回车关闭 PyBullet 窗口并退出... ")
        except KeyboardInterrupt:
            pass
    env.close()

    # Save predictions if requested
    if args.save_preds is not None and len(pred_rows) > 0:
        try:
            import csv
            args.save_preds.parent.mkdir(parents=True, exist_ok=True)
            # collect headers from keys union
            headers = []
            for r in pred_rows:
                for k in r.keys():
                    if k not in headers:
                        headers.append(k)
            with open(args.save_preds, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(pred_rows)
            print(f"Saved predictions to: {args.save_preds}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

    # Save animation if requested and frames available
    if args.render_mode == "rgb_array" and len(frames) > 0 and args.save_animation is not None:
        to_frames_animation(frames, args.save_animation, fps=2)
        print(f"Saved animation: {args.save_animation}")

    # Summary figure
    if args.save_figure is not None:
        plot_summary(path_y, path_z, taken_actions, args.save_figure)
        print(f"Saved summary figure: {args.save_figure}")


if __name__ == "__main__":
    main()
