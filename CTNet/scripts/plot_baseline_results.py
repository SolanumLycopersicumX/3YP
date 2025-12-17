#!/usr/bin/env python3
"""Visualise baseline CSP controller results.

This utility reads the JSON written by `baseline_control.py` and produces
confusion matrices plus trajectory plots for quick diagnostics.

Example
-------
    python scripts/plot_baseline_results.py --subject A01T \
        --output outputs/preprocessing/A01T/baseline_plots.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"


def _load_baseline_json(subject: str) -> dict:
    path = OUTPUT_ROOT / subject / "baseline_control.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline JSON not found: {path}. Run baseline_control.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _plot_confusion(ax, matrix: Iterable[Iterable[int]], labels: List[str], title: str) -> None:
    matrix = np.asarray(matrix, dtype=float)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{int(matrix[i, j])}", ha="center", va="center", color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_trajectory(ax, traj: dict, step_label: str, title: str) -> None:
    y = np.asarray(traj["y"])
    z = np.asarray(traj["z"])
    steps = np.asarray(traj["steps"])

    ax.plot(y, z, marker="o", label=step_label)
    ax.set_xlabel("ΔY (left/right)")
    ax.set_ylabel("ΔZ (up/down)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # annotate step direction for clarity (only for few steps to avoid clutter)
    if len(steps) > 0:
        for idx, (dy, dz) in enumerate(steps[:8], start=1):
            ax.annotate(f"{idx}", (y[idx], z[idx]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot baseline CSP controller results.")
    parser.add_argument("--subject", default="A01T", help="被试编号（默认 A01T）")
    parser.add_argument(
        "--output",
        help="保存图像的路径。若未提供则仅显示。",
    )
    parser.add_argument("--show", action="store_true", help="显示图像（默认仅保存）。")
    args = parser.parse_args()

    data = _load_baseline_json(args.subject)
    metrics = data.get("metrics", {})
    class_order = data.get("class_order", [])
    trajectory = data.get("trajectory", {})

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Baseline Controller Summary - Subject {args.subject}", fontsize=16)

    if "cross_val" in metrics:
        cv = metrics["cross_val"]
        _plot_confusion(axes[0, 0], cv["confusion_matrix"], class_order, f"Cross-val Confusion (acc={cv['accuracy']:.3f})")
    else:
        axes[0, 0].axis("off")
        axes[0, 0].set_title("Cross-val metrics not available")

    if "full_fit" in metrics:
        full = metrics["full_fit"]
        _plot_confusion(axes[0, 1], full["confusion_matrix"], class_order, f"Full-fit Confusion (acc={full['accuracy']:.3f})")
    else:
        axes[0, 1].axis("off")
        axes[0, 1].set_title("Full-fit metrics not available")

    pred_traj = trajectory.get("predicted", {})
    gt_traj = trajectory.get("ground_truth", {})
    if pred_traj:
        ax = axes[1, 0]
        _plot_trajectory(ax, pred_traj, "Predicted", "Predicted trajectory")
    else:
        axes[1, 0].axis("off")
    if gt_traj:
        ax = axes[1, 1]
        _plot_trajectory(ax, gt_traj, "Ground truth", "Ground truth trajectory")
    else:
        axes[1, 1].axis("off")

    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Baseline plots saved to: {out_path}")
    if args.show or not args.output:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
