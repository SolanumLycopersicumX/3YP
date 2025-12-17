import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

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

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from CTNet_model import EEGTransformer
from scipy.io import loadmat


# Deterministic class->movement mapping used in the original control script.
MOVE_MAP = {
    # 坐标约定：x+ 前, y+ 左, z+ 上
    0: (1.0, 0.0),    # left  -> +Y（左）
    1: (-1.0, 0.0),   # right -> -Y（右）
    2: (0.0, 1.0),    # up    -> +Z（上）
    3: (0.0, -1.0),   # down  -> -Z（下）
}


@dataclass
class TrialResult:
    index: int
    prediction: int
    label: int
    dy_pred: float
    dz_pred: float
    dy_true: float
    dz_true: float
    elapsed: float
    y_position: float
    z_position: float


def number_class_channel(database_type: str) -> Tuple[int, int]:
    if database_type == "A":
        return 4, 22
    if database_type == "B":
        return 2, 3
    raise ValueError(f"Unsupported dataset type: {database_type}")


def _resolve_data_path(dir_path: Path, dataset_type: str, subject_id: int, mode: str) -> Path:
    suffix = "T" if mode == "train" else "E"
    return dir_path / f"{dataset_type}{subject_id:02d}{suffix}.mat"


def load_data(dir_path: Path, dataset_type: str, subject_id: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    mat_path = _resolve_data_path(dir_path, dataset_type, subject_id, mode)
    if not mat_path.exists():
        raise FileNotFoundError(f"Missing data file: {mat_path}")
    data_mat = loadmat(mat_path)
    data = data_mat["data"]
    labels = data_mat["label"]
    return data, labels


def load_data_subject_dependent(
    dir_path: Path,
    dataset_type: str,
    subject_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data, train_label = load_data(dir_path, dataset_type, subject_id, mode="train")
    test_data, test_label = load_data(dir_path, dataset_type, subject_id, mode="test")
    return train_data, train_label, test_data, test_label


def load_data_loso(
    dir_path: Path,
    dataset_type: str,
    subject_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, y_train = None, None
    x_test, y_test = None, None

    for idx in range(1, 10):
        x1, y1 = load_data(dir_path, dataset_type, idx, mode="train")
        x2, y2 = load_data(dir_path, dataset_type, idx, mode="test")
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        if idx == subject_id:
            x_test, y_test = x, y
        elif x_train is None:
            x_train, y_train = x, y
        else:
            x_train = np.concatenate((x_train, x), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)

    if x_test is None or y_test is None:
        raise ValueError(f"Subject {subject_id} not found while performing LOSO split.")

    return x_train, y_train, x_test, y_test


def load_data_evaluate(
    dir_path: Path,
    dataset_type: str,
    subject_id: int,
    evaluate_mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if evaluate_mode == "LOSO":
        return load_data_loso(dir_path, dataset_type, subject_id)
    return load_data_subject_dependent(dir_path, dataset_type, subject_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the CTNet-based robotic arm control loop using the "
            "movement mapping from control.py (original literature method)."
        )
    )
    parser.add_argument("--subject", type=int, default=2, help="Subject index (1-9).")
    parser.add_argument("--dataset", choices=["A", "B"], default="A", help="Dataset subset.")
    parser.add_argument(
        "--evaluate-mode",
        choices=["LOSO", "subject-dependent"],
        default="subject-dependent",
        help="Evaluation mode identical to utils.load_data_evaluate.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained CTNet model. Defaults to models/new/model_<subject>.pth",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"), help="Root directory for .mat files.")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device to run inference on.")
    parser.add_argument("--num-trials", type=int, default=10, help="How many test trials to replay.")
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.03,
        help="Physical displacement (meters/radians) attached to a single class decision.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Optional per-trial sleep to mimic real-time inference (seconds).",
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip all visualisations (useful for headless batch runs).",
    )
    parser.add_argument(
        "--save-animation",
        type=Path,
        default=None,
        help="Optional path to save a GIF animation of the predicted motion.",
    )
    parser.add_argument(
        "--save-figure",
        type=Path,
        default=None,
        help="Optional path to save the static diagnostic plot (PNG by default).",
    )
    return parser.parse_args()


def expand_labels(labels: Sequence[int]) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    if labels.size == 0:
        return labels
    if labels.min() == 1:
        labels = labels - 1
    return labels.astype(int)


def load_model(dataset_type: str, subject_id: int, device: torch.device, model_path: Path | None) -> EEGTransformer:
    if model_path is None:
        model_path = Path("./models/new") / f"model_{subject_id}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model checkpoint at {model_path}")

    _, num_channels = number_class_channel(dataset_type)
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
        eeg1_number_channel=num_channels,
        flatten_eeg1=240,
    ).to(device)

    # Robust load to support PyTorch >=2.6 weight-loading behavior
    try:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    if hasattr(state_dict, "state_dict"):
        state_dict = state_dict.state_dict()
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_test_tensor(
    data_dir: Path,
    dataset_type: str,
    subject_id: int,
    evaluate_mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, np.ndarray]:
    _, _, test_data, test_labels = load_data_evaluate(
        data_dir,
        dataset_type,
        subject_id,
        evaluate_mode,
    )
    test_data = np.expand_dims(test_data, axis=1)
    mean = np.mean(test_data)
    std = np.std(test_data) if np.std(test_data) > 0 else 1.0
    test_data = (test_data - mean) / std

    test_tensor = torch.tensor(test_data, dtype=torch.float32, device=device)
    labels = expand_labels(test_labels)
    return test_tensor, labels


def scaled_move(class_id: int, step_size: float) -> Tuple[float, float]:
    base = MOVE_MAP.get(int(class_id), (0.0, 0.0))
    return base[0] * step_size, base[1] * step_size


def run_trials(
    model: EEGTransformer,
    test_tensor: torch.Tensor,
    labels: np.ndarray,
    step_size: float,
    num_trials: int,
    sleep_time: float,
) -> Tuple[List[TrialResult], List[Tuple[float, float]], List[Tuple[float, float]]]:
    total_trials = min(num_trials, len(test_tensor))
    results: List[TrialResult] = []
    pred_path: List[Tuple[float, float]] = [(0.0, 0.0)]
    true_path: List[Tuple[float, float]] = [(0.0, 0.0)]
    y, z = 0.0, 0.0

    for idx in range(total_trials):
        x_input = test_tensor[idx].unsqueeze(0)  # shape [1, 1, 22, 1000]
        start = time.time()
        with torch.no_grad():
            logits = model(x_input)
            if isinstance(logits, tuple):
                _, logits = logits
            pred_cls = torch.argmax(logits, dim=1).item()
        elapsed = time.time() - start

        true_cls = int(labels[idx])
        dy_pred, dz_pred = scaled_move(pred_cls, step_size)
        dy_true, dz_true = scaled_move(true_cls, step_size)

        y += dy_pred
        z += dz_pred
        pred_path.append((y, z))
        last_true_y, last_true_z = true_path[-1]
        true_path.append((last_true_y + dy_true, last_true_z + dz_true))

        results.append(
            TrialResult(
                index=idx,
                prediction=pred_cls,
                label=true_cls,
                dy_pred=dy_pred,
                dz_pred=dz_pred,
                dy_true=dy_true,
                dz_true=dz_true,
                elapsed=elapsed,
                y_position=y,
                z_position=z,
            )
        )

        print(
            f"[Trial {idx + 1}] pred={pred_cls} true={true_cls} "
            f"Δy/Δz pred=({dy_pred:.3f},{dz_pred:.3f}) true=({dy_true:.3f},{dz_true:.3f}) "
            f"pose=(y={y:.3f}, z={z:.3f}) time={elapsed:.3f}s"
        )

        if sleep_time > 0:
            time.sleep(sleep_time)

    return results, pred_path, true_path


def draw_static_diagnostics(
    pred_path: Sequence[Tuple[float, float]],
    true_path: Sequence[Tuple[float, float]],
    results: Sequence[TrialResult],
    save_path: Path | None,
) -> None:
    pred_y = np.array([y for y, _ in pred_path])
    pred_z = np.array([z for _, z in pred_path])
    true_y = np.array([y for y, _ in true_path])
    true_z = np.array([z for _, z in true_path])

    pred_dy = np.array([r.dy_pred for r in results])
    pred_dz = np.array([r.dz_pred for r in results])
    true_dy = np.array([r.dy_true for r in results])
    true_dz = np.array([r.dz_true for r in results])

    pos_x = np.arange(len(pred_path))
    diff_x = np.arange(1, len(pred_path))

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    axs[0].plot(pos_x, pred_y, marker="o", label="Predicted Y")
    axs[0].plot(pos_x, true_y, marker="x", label="True Y")
    axs[0].set_title("Y position (cumulative)")
    axs[0].set_ylabel("Y")
    axs[0].grid(alpha=0.3)
    axs[0].legend()

    axs[1].plot(pos_x, pred_z, marker="o", label="Predicted Z")
    axs[1].plot(pos_x, true_z, marker="x", label="True Z")
    axs[1].set_title("Z position (cumulative)")
    axs[1].set_ylabel("Z")
    axs[1].grid(alpha=0.3)
    axs[1].legend()

    axs[2].plot(diff_x, pred_dy, marker="o", label="Predicted ΔY")
    axs[2].plot(diff_x, true_dy, marker="x", label="True ΔY")
    axs[2].set_title("ΔY per trial")
    axs[2].set_ylabel("ΔY")
    axs[2].grid(alpha=0.3)
    axs[2].legend()

    axs[3].plot(diff_x, pred_dz, marker="o", label="Predicted ΔZ")
    axs[3].plot(diff_x, true_dz, marker="x", label="True ΔZ")
    axs[3].set_title("ΔZ per trial")
    axs[3].set_xlabel("Trial")
    axs[3].set_ylabel("ΔZ")
    axs[3].grid(alpha=0.3)
    axs[3].legend()

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Saved static diagnostic plot to {save_path}")

    plt.show()


def animate_path(
    pred_path: Sequence[Tuple[float, float]],
    true_path: Sequence[Tuple[float, float]],
    save_path: Path | None,
) -> None:
    pred_y = np.array([y for y, _ in pred_path])
    pred_z = np.array([z for _, z in pred_path])
    true_y = np.array([y for y, _ in true_path])
    true_z = np.array([z for _, z in true_path])

    min_y = min(pred_y.min(), true_y.min()) - 0.05
    max_y = max(pred_y.max(), true_y.max()) + 0.05
    min_z = min(pred_z.min(), true_z.min()) - 0.05
    max_z = max(pred_z.max(), true_z.max()) + 0.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Simulated end-effector trajectory (YZ plane)")
    ax.set_xlabel("Y (lateral)")
    ax.set_ylabel("Z (vertical)")
    ax.grid(alpha=0.3)
    ax.set_xlim(min_y, max_y)
    ax.set_ylim(min_z, max_z)

    pred_line, = ax.plot([], [], "-o", color="#1f77b4", label="Predicted path")
    true_line, = ax.plot([], [], "--x", color="#ff7f0e", label="Reference path")
    pred_marker, = ax.plot([], [], "o", color="#1f77b4", markersize=10)
    true_marker, = ax.plot([], [], "x", color="#ff7f0e", markersize=10)
    ax.legend()

    def init():
        pred_line.set_data([], [])
        true_line.set_data([], [])
        pred_marker.set_data([], [])
        true_marker.set_data([], [])
        return pred_line, true_line, pred_marker, true_marker

    def update(frame: int):
        pred_line.set_data(pred_y[: frame + 1], pred_z[: frame + 1])
        true_line.set_data(true_y[: frame + 1], true_z[: frame + 1])
        pred_marker.set_data(pred_y[frame], pred_z[frame])
        true_marker.set_data(true_y[frame], true_z[frame])
        return pred_line, true_line, pred_marker, true_marker

    anim = FuncAnimation(fig, update, frames=len(pred_path), init_func=init, interval=800, blit=True, repeat=False)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(save_path, writer=PillowWriter(fps=2))
        print(f"Saved trajectory animation to {save_path}")

    plt.show()


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args.dataset, args.subject, device, args.model_path)
    test_tensor, labels = load_test_tensor(args.data_dir, args.dataset, args.subject, args.evaluate_mode, device)

    results, pred_path, true_path = run_trials(model, test_tensor, labels, args.step_size, args.num_trials, args.sleep)

    if not results:
        print("No trials were processed. Check dataset/model configuration.")
        return

    predictions = np.array([r.prediction for r in results])
    truths = np.array([r.label for r in results])
    accuracy = (predictions == truths).mean()

    print("\n--- Summary ---")
    print(f"Trials processed: {len(results)}")
    print(f"Accuracy: {accuracy * 100:.2f}% ({(predictions == truths).sum()}/{len(results)})")
    print(f"Average inference time per trial: {np.mean([r.elapsed for r in results]):.3f}s")

    if args.no_figure:
        return

    if args.save_animation is not None or not args.no_figure:
        animate_path(pred_path, true_path, args.save_animation)

    draw_static_diagnostics(pred_path, true_path, results, args.save_figure)


if __name__ == "__main__":
    main()
