#!/usr/bin/env python3
"""
ICA Artifact Removal Comparison Experiment

Compares CTNet classification accuracy with and without ICA preprocessing
on BCI Competition IV-2a dataset (MAT format).

Uses sklearn FastICA on epoched data since GDF reading has MNE version
compatibility issues. Applies per-trial ICA to remove the highest-variance
independent components (a proxy for ocular/muscular artefacts).

Usage:
  python scripts/ica_comparison_run.py
  python scripts/ica_comparison_run.py --subjects 1 3 7 --epochs 200
"""

from __future__ import annotations
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("WARNING: sklearn required for ICA. Run: pip install scikit-learn")

DATA_ROOT = _ROOT / "mymat_raw"
OUTPUT_DIR = _ROOT / "outputs" / "ica_comparison"


# ============================================================================
# MAT-based Preprocessing
# ============================================================================

def load_mat_data(subject_id: int, mode: str = "T") -> Tuple[np.ndarray, np.ndarray]:
    mat_path = DATA_ROOT / f"A{subject_id:02d}{mode}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    mat = scipy.io.loadmat(str(mat_path))
    data = mat["data"].astype(np.float64)
    labels = mat["label"].flatten().astype(np.int64) - 1
    return data, labels


def bandpass_filter(data: np.ndarray, low: float = 8.0, high: float = 30.0,
                    fs: float = 250.0, order: int = 4) -> np.ndarray:
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[i, ch] = filtfilt(b, a, data[i, ch])
    return filtered


def apply_ica_artifact_removal(data: np.ndarray, n_components: int = 20,
                               n_artifacts: int = 2) -> Tuple[np.ndarray, Dict]:
    """Apply FastICA per trial and remove highest-variance components."""
    n_trials, n_ch, n_times = data.shape
    n_components = min(n_components, n_ch)
    cleaned = np.zeros_like(data)
    total_removed = 0

    for i in range(n_trials):
        trial = data[i]
        try:
            ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
            sources = ica.fit_transform(trial.T)
            variances = np.var(sources, axis=0)
            artifact_idx = np.argsort(variances)[-n_artifacts:]
            sources[:, artifact_idx] = 0
            cleaned[i] = ica.inverse_transform(sources).T
            total_removed += n_artifacts
        except Exception:
            cleaned[i] = trial

    info = {"n_components": n_components, "n_artifacts_per_trial": n_artifacts,
            "total_removed": total_removed, "n_trials": n_trials}
    return cleaned, info


# ============================================================================
# CTNet Training (lightweight per-subject eval)
# ============================================================================

def train_and_eval_ctnet(
    train_data: np.ndarray, train_labels: np.ndarray,
    test_data: np.ndarray, test_labels: np.ndarray,
    n_channels: int = 22, n_classes: int = 4,
    epochs: int = 200, lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict:
    from CTNet_model import EEGTransformer

    n_times = train_data.shape[2]
    pool1, pool2 = 8, 8
    flatten_size = 40 * (n_times // (pool1 * pool2))

    model = EEGTransformer(
        number_class=n_classes, number_channel=n_channels,
        flatten_eeg1=flatten_size,
        eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2, eeg1_dropout_rate=0.3,
        pooling_size1=pool1, pooling_size2=pool2,
        heads=2, depth=6, emb_size=40,
    ).to(device)

    X_train = np.expand_dims(train_data, axis=1)
    X_test = np.expand_dims(test_data, axis=1)

    mean_val = X_train.mean()
    std_val = X_train.std()
    X_train = (X_train - mean_val) / (std_val + 1e-12)
    X_test = (X_test - mean_val) / (std_val + 1e-12)

    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(train_labels).long(),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(),
        torch.from_numpy(test_labels).long(),
    )
    train_loader = DataLoader(train_ds, batch_size=72, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=72, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            _, logits = model(X_batch)
            loss = criterion(logits, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    _, logits = model(X_batch)
                    correct += (logits.argmax(dim=-1) == y_batch).sum().item()
                    total += len(y_batch)
            acc = correct / total * 100
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= 10:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_labels_list = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            _, logits = model(X_batch)
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_labels_list.append(y_batch.numpy())
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_labels_list)
    final_acc = (preds == true).mean() * 100

    per_class = {}
    for c in range(n_classes):
        mask = true == c
        if mask.sum() > 0:
            per_class[c] = (preds[mask] == true[mask]).mean() * 100

    return {"accuracy": round(best_acc, 2), "final_accuracy": round(final_acc, 2),
            "per_class": per_class, "n_train": len(train_labels), "n_test": len(test_labels)}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="ICA Comparison Experiment")
    parser.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        print(f"ERROR: MAT data directory not found: {DATA_ROOT}")
        return

    results = {"no_ica": {}, "with_ica": {}, "comparison": {}}

    for subject_id in args.subjects:
        train_mat = DATA_ROOT / f"A{subject_id:02d}T.mat"
        test_mat = DATA_ROOT / f"A{subject_id:02d}E.mat"

        if not train_mat.exists() or not test_mat.exists():
            print(f"  S{subject_id}: MAT files missing, skipping")
            continue

        try:
            train_data_raw, train_labels = load_mat_data(subject_id, "T")
            test_data_raw, test_labels = load_mat_data(subject_id, "E")
        except Exception as e:
            print(f"  S{subject_id}: Load error: {e}")
            continue

        for condition, use_ica in [("no_ica", False), ("with_ica", True)]:
            label = "with ICA" if use_ica else "without ICA"
            print(f"\n{'='*60}")
            print(f"Subject {subject_id} - {label}")
            print(f"{'='*60}")

            try:
                print(f"  Bandpass filtering 8-30 Hz...")
                train_data = bandpass_filter(train_data_raw.copy())
                test_data = bandpass_filter(test_data_raw.copy())

                ica_info = {"n_artifacts_per_trial": 0}
                if use_ica:
                    print(f"  Applying FastICA artifact removal...")
                    train_data, ica_train = apply_ica_artifact_removal(train_data)
                    test_data, ica_test = apply_ica_artifact_removal(test_data)
                    ica_info = ica_train
                    print(f"    Removed {ica_train['n_artifacts_per_trial']} components/trial")

                print(f"  Training CTNet ({args.epochs} epochs)...")
                res = train_and_eval_ctnet(
                    train_data.astype(np.float32), train_labels,
                    test_data.astype(np.float32), test_labels,
                    n_channels=train_data.shape[1], epochs=args.epochs, device=device,
                )
                print(f"  => Best accuracy: {res['accuracy']:.2f}%")

                results[condition][subject_id] = {
                    "accuracy": res["accuracy"],
                    "per_class": {str(k): v for k, v in res["per_class"].items()},
                    "n_train": res["n_train"],
                    "n_test": res["n_test"],
                    "ica_artifacts_removed": ica_info.get("n_artifacts_per_trial", 0),
                }
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()

    # Comparison
    print(f"\n{'='*60}")
    print("ICA Comparison Results")
    print(f"{'='*60}")

    print(f"\n{'Subject':<10} {'No ICA':>10} {'With ICA':>10} {'Diff':>10}")
    print("-" * 45)

    improvements = []
    for sid in args.subjects:
        no_ica = results["no_ica"].get(sid, {}).get("accuracy")
        with_ica = results["with_ica"].get(sid, {}).get("accuracy")
        if no_ica is not None and with_ica is not None:
            diff = with_ica - no_ica
            improvements.append(diff)
            sign = "+" if diff > 0 else ""
            print(f"S{sid:<8} {no_ica:>9.2f}% {with_ica:>9.2f}% {sign}{diff:>9.2f}%")
            results["comparison"][str(sid)] = {
                "no_ica": no_ica, "with_ica": with_ica, "diff": round(diff, 2),
            }

    if improvements:
        avg_no = np.mean([results["no_ica"][s]["accuracy"] for s in args.subjects if s in results["no_ica"]])
        avg_with = np.mean([results["with_ica"][s]["accuracy"] for s in args.subjects if s in results["with_ica"]])
        avg_diff = avg_with - avg_no
        print("-" * 45)
        sign = "+" if avg_diff > 0 else ""
        print(f"{'Average':<10} {avg_no:>9.2f}% {avg_with:>9.2f}% {sign}{avg_diff:>9.2f}%")

        print(f"\nSummary:")
        print(f"  Improved with ICA: {sum(1 for d in improvements if d > 0)}/{len(improvements)}")
        print(f"  Degraded with ICA: {sum(1 for d in improvements if d < 0)}/{len(improvements)}")
        print(f"  Average change: {np.mean(improvements):+.2f}%")

        results["summary"] = {
            "avg_no_ica": round(avg_no, 2),
            "avg_with_ica": round(avg_with, 2),
            "avg_improvement": round(avg_diff, 2),
            "n_improved": sum(1 for d in improvements if d > 0),
            "n_degraded": sum(1 for d in improvements if d < 0),
            "n_subjects": len(improvements),
            "timestamp": datetime.now().isoformat(),
        }

    with open(OUTPUT_DIR / "ica_comparison_result.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {OUTPUT_DIR / 'ica_comparison_result.json'}")


if __name__ == "__main__":
    main()
