#!/usr/bin/env python3
"""Baseline controller that maps CSP/LDA predictions to discrete robot actions.

This script provides a non-RL baseline for the Phase 3 control task. It
reuses the CSP log-variance features exported during the preprocessing
stage, fits a Linear Discriminant Analysis (LDA) classifier, predicts motor
imagery classes for each epoch, and converts them into
trajectory commands (left/right/up/down) using a rule-based mapping.

The resulting accuracy, confusion matrix, and simulated trajectories are
saved under `outputs/preprocessing/<subject>/baseline_control.json` for
easy comparison with the RL controller.

Example
-------
    python scripts/baseline_control.py --subject A01T --step-size 0.03
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold


class SimpleGaussianNB:
    """Gaussian Naive Bayes classifier implemented with NumPy only."""

    def __init__(self, var_smoothing: float = 1e-6) -> None:
        self.var_smoothing = var_smoothing
        self.classes_: np.ndarray | None = None
        self.theta_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SimpleGaussianNB":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        self.classes_ = classes
        n_samples, n_features = X.shape
        theta = np.zeros((len(classes), n_features), dtype=np.float64)
        sigma = np.zeros_like(theta)
        for idx, cls in enumerate(classes):
            Xi = X[y == cls]
            theta[idx] = Xi.mean(axis=0)
            sigma[idx] = Xi.var(axis=0) + self.var_smoothing
        priors = counts / n_samples
        self.theta_ = theta
        self.sigma_ = sigma
        self.class_log_prior_ = np.log(priors)
        return self

    def _log_gaussian_prob(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None or self.sigma_ is None or self.class_log_prior_ is None:
            raise RuntimeError("SimpleGaussianNB must be fitted before prediction.")
        X = np.asarray(X, dtype=np.float64)
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * self.sigma_), axis=1)
        log_prob = log_prob + self.class_log_prior_
        diff = X[:, None, :] - self.theta_[None, :, :]
        log_prob = log_prob + -0.5 * np.sum((diff**2) / self.sigma_[None, :, :], axis=2)
        return log_prob

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._log_gaussian_prob(X)
        assert self.classes_ is not None
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._log_gaussian_prob(X)
        scores -= scores.max(axis=1, keepdims=True)
        exp = np.exp(scores)
        prob = exp / exp.sum(axis=1, keepdims=True)
        return prob

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "preprocessing"

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

DEFAULT_ACTION_MAP: Dict[str, Tuple[float, float]] = {
    # dy, dz (left/right, up/down)
    "left_hand": (-1.0, 0.0),
    "right_hand": (1.0, 0.0),
    "foot": (0.0, -1.0),
    "tongue": (0.0, 1.0),
}


def load_csp_artifacts(subject: str):
    subject_dir = OUTPUT_ROOT / subject
    feature_path = subject_dir / "csp_features.npz"
    metadata_path = subject_dir / "csp_metadata.json"
    if not feature_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing CSP artifacts for subject {subject}. Run preprocessing first."
        )
    npz = np.load(feature_path)
    features = npz["features"].astype(np.float32)
    labels = npz["labels"].astype(int)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return features, labels, metadata


def parse_action_map(args: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    mapping: Dict[str, Tuple[float, float]] = {}
    for entry in args:
        if "=" not in entry:
            raise ValueError(f"Invalid mapping entry '{entry}'. Use format event=dy,dz")
        name, coords = entry.split("=", 1)
        parts = coords.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid coordinate format '{coords}'. Expected dy,dz")
        try:
            dy = float(parts[0])
            dz = float(parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid numeric values in '{coords}'") from exc
        mapping[name.strip()] = (dy, dz)
    return mapping


def build_action_lookup(
    metadata: Dict[str, object],
    user_map: Dict[str, Tuple[float, float]],
    step_size: float,
) -> Dict[int, Tuple[float, float]]:
    event_id = metadata.get("event_id", {})
    if not isinstance(event_id, dict):
        raise ValueError("Metadata missing 'event_id' mapping. Re-run OVR CSP pipeline.")

    action_map = DEFAULT_ACTION_MAP.copy()
    action_map.update(user_map)

    lookup: Dict[int, Tuple[float, float]] = {}
    for name, code in event_id.items():
        base = action_map.get(name)
        if base is None:
            raise ValueError(
                f"No action mapping provided for event '{name}'. "
                f"Use --action-map {name}=dy,dz to specify it."
            )
        dy, dz = base
        lookup[int(code)] = (dy * step_size, dz * step_size)
    return lookup


def simulate_trajectory(
    labels: np.ndarray,
    action_lookup: Dict[int, Tuple[float, float]],
    *,
    origin: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_pos = [origin[0]]
    z_pos = [origin[1]]
    steps = []
    y, z = origin
    for label in labels:
        dy, dz = action_lookup.get(int(label), (0.0, 0.0))
        y += dy
        z += dz
        y_pos.append(y)
        z_pos.append(z)
        steps.append((dy, dz))
    return np.asarray(y_pos), np.asarray(z_pos), np.asarray(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline CSP/LDA controller.")
    parser.add_argument("--subject", default="A01T", help="被试编号（默认 A01T）")
    parser.add_argument("--step-size", type=float, default=0.03, help="单步位移幅度")
    parser.add_argument(
        "--action-map",
        action="append",
        default=[],
        help="自定义事件到动作的映射，格式 event=dy,dz，可重复使用。",
    )
    parser.add_argument(
        "--output",
        help="结果写入的 JSON 路径，默认 subject 目录下 baseline_control.json",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--random-state", type=int, default=42, help="交叉验证随机种子")
    parser.add_argument("--var-smoothing", type=float, default=1e-6, help="高斯朴素贝叶斯方差平滑值")
    args = parser.parse_args()

    features, labels, metadata = load_csp_artifacts(args.subject)
    subject_dir = OUTPUT_ROOT / args.subject

    user_map = parse_action_map(args.action_map)
    action_lookup = build_action_lookup(metadata, user_map, args.step_size)

    cv = StratifiedKFold(n_splits=args.cv_splits, shuffle=True, random_state=args.random_state)
    cv_preds = np.zeros_like(labels)
    for train_idx, test_idx in cv.split(features, labels):
        cv_model = SimpleGaussianNB(var_smoothing=args.var_smoothing)
        cv_model.fit(features[train_idx], labels[train_idx])
        cv_preds[test_idx] = cv_model.predict(features[test_idx])

    cv_accuracy = accuracy_score(labels, cv_preds)
    class_order = sorted(action_lookup.keys())
    cv_conf = confusion_matrix(labels, cv_preds, labels=class_order)
    cv_report = classification_report(labels, cv_preds, labels=class_order, output_dict=True)

    final_model = SimpleGaussianNB(var_smoothing=args.var_smoothing)
    final_model.fit(features, labels)
    preds = final_model.predict(features)
    full_accuracy = accuracy_score(labels, preds)
    full_conf = confusion_matrix(labels, preds, labels=class_order)
    full_report = classification_report(labels, preds, labels=class_order, output_dict=True)

    id_to_name = {int(v): k for k, v in metadata.get("event_id", {}).items()}
    pred_names = [id_to_name.get(int(p), str(p)) for p in preds]
    true_names = [id_to_name.get(int(t), str(t)) for t in labels]

    pred_y, pred_z, pred_steps = simulate_trajectory(preds, action_lookup)
    true_y, true_z, true_steps = simulate_trajectory(labels, action_lookup)

    results = {
        "subject": args.subject,
        "step_size": args.step_size,
        "action_lookup": {
            id_to_name.get(code, str(code)): {"dy": dy, "dz": dz}
            for code, (dy, dz) in action_lookup.items()
        },
        "metrics": {
            "cross_val": {
                "splits": args.cv_splits,
                "random_state": args.random_state,
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
        "class_order": [id_to_name.get(code, str(code)) for code in class_order],
        "var_smoothing": args.var_smoothing,
        "predicted_labels": pred_names,
        "true_labels": true_names,
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

    output_path = Path(args.output) if args.output else subject_dir / "baseline_control.json"
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Baseline] subject={args.subject} cv_accuracy={cv_accuracy:.4f} full_fit={full_accuracy:.4f}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
