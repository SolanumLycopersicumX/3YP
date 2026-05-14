#!/usr/bin/env python3
"""Run the CSP baseline multiple times and aggregate metrics/plots.

This helper reuses the logic from `baseline_control.py` (SimpleGaussianNB)
to execute the baseline pipeline with different random seeds, then reports
the averaged accuracy, confusion matrices (mean) and trajectories.

Example
-------
    python scripts/baseline_batch.py --subject A01T --runs 5 \
        --seeds 0 1 2 3 4 \
        --output outputs/preprocessing/A01T/baseline_batch.json \
        --plot   outputs/preprocessing/A01T/baseline_batch.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from baseline_control import (
    load_csp_artifacts,
    parse_action_map,
    build_action_lookup,
    simulate_trajectory,
    SimpleGaussianNB,
    OUTPUT_ROOT,
    PROJECT_ROOT,
)


def _run_single(
    features: np.ndarray,
    labels: np.ndarray,
    metadata: Dict[str, object],
    *,
    step_size: float,
    var_smoothing: float,
    cv_splits: int,
    seed: int,
) -> Dict[str, object]:
    action_lookup = build_action_lookup(metadata, {}, step_size)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    cv_preds = np.zeros_like(labels)
    for train_idx, test_idx in cv.split(features, labels):
        model = SimpleGaussianNB(var_smoothing=var_smoothing)
        model.fit(features[train_idx], labels[train_idx])
        cv_preds[test_idx] = model.predict(features[test_idx])

    cv_accuracy = accuracy_score(labels, cv_preds)
    class_order = sorted(action_lookup.keys())
    id_to_name = {int(v): k for k, v in metadata.get("event_id", {}).items()}

    cv_conf = confusion_matrix(labels, cv_preds, labels=class_order)
    cv_report = classification_report(labels, cv_preds, labels=class_order, output_dict=True)

    final_model = SimpleGaussianNB(var_smoothing=var_smoothing)
    final_model.fit(features, labels)
    preds = final_model.predict(features)
    full_accuracy = accuracy_score(labels, preds)
    full_conf = confusion_matrix(labels, preds, labels=class_order)
    full_report = classification_report(labels, preds, labels=class_order, output_dict=True)

    pred_y, pred_z, pred_steps = simulate_trajectory(preds, action_lookup)
    true_y, true_z, true_steps = simulate_trajectory(labels, action_lookup)

    return {
        "seed": seed,
        "metrics": {
            "cross_val": {
                "accuracy": float(cv_accuracy),
                "confusion_matrix": cv_conf.tolist(),
                "classification_report": cv_report,
            },
            "full_fit": {
                "accuracy": float(full_accuracy),
                "confusion_matrix": full_conf.tolist(),
                "classification_report": full_report,
            },
        },
        "predicted_labels": [id_to_name.get(int(p), str(p)) for p in preds],
        "trajectory": {
            "predicted": {
                "y": pred_y.tolist(),
                "z": pred_z.tolist(),
                "steps": pred_steps.tolist(),
            },
            "ground_truth": {
                "y": true_y.tolist(),
                "z": true_z.tolist(),
                "steps": true_steps.tolist(),
            },
        },
    }


def _avg_confusion(mats: List[np.ndarray]) -> np.ndarray:
    return np.mean(np.stack(mats, axis=0), axis=0)


def _plot_summary(
    subject: str,
    results: Dict[str, object],
    class_labels: List[str],
    output_path: Path | None,
    show: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Baseline Batch Summary - {subject}", fontsize=16)

    cv_conf = np.array(results["average"]["metrics"]["cross_val"]["confusion_matrix"])
    full_conf = np.array(results["average"]["metrics"]["full_fit"]["confusion_matrix"])

    def plot_conf(ax, matrix, title):
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(class_labels)), class_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(class_labels)), class_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_conf(
        axes[0, 0],
        cv_conf,
        f"Cross-val (avg acc={results['average']['metrics']['cross_val']['accuracy']:.3f})",
    )
    plot_conf(
        axes[0, 1],
        full_conf,
        f"Full-fit (avg acc={results['average']['metrics']['full_fit']['accuracy']:.3f})",
    )

    avg_pred = results["average"]["trajectory"]["predicted"]
    avg_gt = results["average"]["trajectory"]["ground_truth"]

    def plot_traj(ax, traj, label, title):
        y = np.asarray(traj["y"])
        z = np.asarray(traj["z"])
        ax.plot(y, z, marker="o", label=label)
        ax.set_xlabel("ΔY (left/right)")
        ax.set_ylabel("ΔZ (up/down)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plot_traj(axes[1, 0], avg_pred, "Predicted", "Avg predicted trajectory")
    plot_traj(axes[1, 1], avg_gt, "Ground truth", "Avg ground truth trajectory")

    axes[1, 0].legend()
    axes[1, 1].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Batch plot saved to: {output_path}")
    if show or not output_path:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate baseline results over multiple runs.")
    parser.add_argument("--subject", default="A01T", help="被试编号（默认 A01T）")
    parser.add_argument("--runs", type=int, default=5, help="运行次数")
    parser.add_argument("--seeds", nargs="*", type=int, help="指定 seeds（长度需 >= runs）")
    parser.add_argument("--step-size", type=float, default=0.03, help="单步位移幅度")
    parser.add_argument("--var-smoothing", type=float, default=1e-6, help="高斯朴素贝叶斯方差平滑值")
    parser.add_argument("--cv-splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--output", help="保存聚合结果 JSON 的路径")
    parser.add_argument("--plot", help="保存平均指标图的路径")
    parser.add_argument("--show", action="store_true", help="显示图像（默认不显示）")
    args = parser.parse_args()

    features, labels, metadata = load_csp_artifacts(args.subject)
    class_order = sorted(build_action_lookup(metadata, {}, args.step_size).keys())
    id_to_name = {int(v): k for k, v in metadata.get("event_id", {}).items()}
    class_labels = [id_to_name.get(code, str(code)) for code in class_order]

    seeds = args.seeds if args.seeds else list(range(args.runs))
    if len(seeds) < args.runs:
        raise ValueError("Provide at least `runs` seeds or let script auto-generate.")

    runs = []
    for idx in range(args.runs):
        seed = seeds[idx]
        print(f"[Batch] Running baseline #{idx+1}/{args.runs} with seed={seed}")
        result = _run_single(
            features,
            labels,
            metadata,
            step_size=args.step_size,
            var_smoothing=args.var_smoothing,
            cv_splits=args.cv_splits,
            seed=seed,
        )
        runs.append(result)

    # Aggregate metrics
    cv_accs = [run["metrics"]["cross_val"]["accuracy"] for run in runs]
    full_accs = [run["metrics"]["full_fit"]["accuracy"] for run in runs]
    avg_cv_conf = _avg_confusion([np.array(run["metrics"]["cross_val"]["confusion_matrix"]) for run in runs])
    avg_full_conf = _avg_confusion([np.array(run["metrics"]["full_fit"]["confusion_matrix"]) for run in runs])

    # Average trajectories
    pred_ys = np.array([np.asarray(run["trajectory"]["predicted"]["y"]) for run in runs])
    pred_zs = np.array([np.asarray(run["trajectory"]["predicted"]["z"]) for run in runs])
    gt_ys = np.array([np.asarray(run["trajectory"]["ground_truth"]["y"]) for run in runs])
    gt_zs = np.array([np.asarray(run["trajectory"]["ground_truth"]["z"]) for run in runs])
    avg_pred_traj = {"y": pred_ys.mean(axis=0).tolist(), "z": pred_zs.mean(axis=0).tolist()}
    avg_gt_traj = {"y": gt_ys.mean(axis=0).tolist(), "z": gt_zs.mean(axis=0).tolist()}

    aggregate = {
        "subject": args.subject,
        "step_size": args.step_size,
        "var_smoothing": args.var_smoothing,
        "runs": runs,
        "average": {
            "metrics": {
                "cross_val": {
                    "accuracy": float(np.mean(cv_accs)),
                    "std_accuracy": float(np.std(cv_accs)),
                    "confusion_matrix": avg_cv_conf.tolist(),
                },
                "full_fit": {
                    "accuracy": float(np.mean(full_accs)),
                    "std_accuracy": float(np.std(full_accs)),
                    "confusion_matrix": avg_full_conf.tolist(),
                },
            },
            "trajectory": {
                "predicted": avg_pred_traj,
                "ground_truth": avg_gt_traj,
            },
        },
        "class_order": class_labels,
    }

    output_path = Path(args.output) if args.output else OUTPUT_ROOT / args.subject / "baseline_batch.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Batch summary saved to: {output_path}")

    plot_path = Path(args.plot) if args.plot else None
    _plot_summary(args.subject, aggregate, class_labels, plot_path, args.show)


if __name__ == "__main__":
    main()
