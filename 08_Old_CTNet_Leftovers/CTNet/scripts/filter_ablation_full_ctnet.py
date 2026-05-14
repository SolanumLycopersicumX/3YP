#!/usr/bin/env python3
"""
Filter Ablation Study with Full EEGTransformer

This experiment uses the FULL EEGTransformer model (not a simplified version)
to properly evaluate the effect of 8-30Hz bandpass filtering.

Conditions:
  A) With 8-30Hz bandpass filter (current pipeline)
  B) Without filter (raw normalized data)

This will definitively show whether the filter contributes to the model's
high accuracy.
"""

from __future__ import annotations
import argparse, json, sys, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, resample

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

# Import the FULL CTNet model
from CTNet_model import EEGTransformer

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False

PHYSIONET_DIR = _ROOT / "physionet_raw"
OUTPUT_DIR = _ROOT / "outputs" / "filter_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_physionet_subject(subject_id: int, task_runs: List[int] = [4, 8, 12]) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw EEG from PhysioNet."""
    if not MNE_AVAILABLE:
        raise RuntimeError("MNE is required")
    
    all_epochs = []
    all_labels = []
    
    subject_dir = PHYSIONET_DIR / f"S{subject_id:03d}"
    
    for run in task_runs:
        edf_path = subject_dir / f"S{subject_id:03d}R{run:02d}.edf"
        if not edf_path.exists():
            continue
        
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.0,
                           baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data()
        labels = epochs.events[:, 2]
        
        all_epochs.append(data)
        all_labels.append(labels)
    
    if not all_epochs:
        raise ValueError(f"No data found for subject {subject_id}")
    
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    unique_labels = np.unique(y)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y = np.array([label_map[l] for l in y])
    
    return X, y


def bandpass_filter(data: np.ndarray, low: float = 8.0, high: float = 30.0,
                    fs: float = 160.0, order: int = 4) -> np.ndarray:
    """Apply 8-30Hz bandpass filter."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[i, ch] = filtfilt(b, a, data[i, ch])
    return filtered


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Z-score normalization per trial."""
    normalized = np.zeros_like(data)
    for i in range(data.shape[0]):
        mean = data[i].mean()
        std = data[i].std()
        if std > 1e-8:
            normalized[i] = (data[i] - mean) / std
        else:
            normalized[i] = data[i] - mean
    return normalized


def resample_data(data: np.ndarray, target_len: int = 1000) -> np.ndarray:
    """Resample to fixed length (1000 samples as used in training)."""
    n_trials, n_ch, n_times = data.shape
    resampled = np.zeros((n_trials, n_ch, target_len), dtype=np.float32)
    for i in range(n_trials):
        for ch in range(n_ch):
            resampled[i, ch] = resample(data[i, ch], target_len)
    return resampled


def train_and_eval_ctnet(train_data: np.ndarray, train_labels: np.ndarray,
                         test_data: np.ndarray, test_labels: np.ndarray,
                         n_channels: int = 64, n_epochs: int = 100,
                         device: str = "cuda" if torch.cuda.is_available() else "cpu"
                         ) -> Dict:
    """Train FULL EEGTransformer and evaluate."""
    
    # Add channel dimension: (B, C, T) -> (B, 1, C, T)
    X_train = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    n_classes = len(np.unique(train_labels))
    n_times = train_data.shape[2]
    
    # Use the FULL EEGTransformer model
    model = EEGTransformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.3
    )
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    # Training
    model.train()
    best_acc = 0
    for epoch in range(n_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        
        # Quick eval every 20 epochs
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                X_t = X_test.to(device)
                logits = model(X_t)
                preds = logits.argmax(dim=1).cpu().numpy()
                acc = (preds == y_test.numpy()).mean() * 100
                if acc > best_acc:
                    best_acc = acc
            model.train()
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    y_true = y_test.numpy()
    accuracy = (preds == y_true).mean() * 100
    
    return {"accuracy": float(accuracy), "best_accuracy": float(best_acc), "n_test": len(y_true)}


def run_experiment(subjects: List[int], n_epochs: int = 150) -> Dict:
    """Run filter ablation study."""
    
    results = {
        "experiment": "Filter Ablation with Full EEGTransformer",
        "date": datetime.now().isoformat(),
        "model": "EEGTransformer (full, not simplified)",
        "conditions": {
            "A": "With 8-30Hz bandpass filter",
            "B": "Without filter (raw normalized)"
        },
        "subjects": {},
        "summary": {}
    }
    
    all_A, all_B = [], []
    
    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"Subject {subj}")
        print('='*60)
        
        try:
            # Load raw data
            print("Loading raw EEG data...")
            data, labels = load_physionet_subject(subj)
            n_trials = len(labels)
            n_channels = data.shape[1]
            
            print(f"  Data shape: {data.shape}")
            print(f"  Labels: {np.bincount(labels)}")
            
            # Split train/test (80/20)
            n_train = int(n_trials * 0.8)
            indices = np.random.RandomState(42).permutation(n_trials)
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            
            subj_results = {}
            
            # ----- Condition A: With Filter -----
            print("\n[A] With 8-30Hz Bandpass Filter...")
            data_A = bandpass_filter(data.copy())
            data_A = normalize_data(data_A)
            data_A = resample_data(data_A, target_len=1000)
            
            metrics_A = train_and_eval_ctnet(
                data_A[train_idx], labels[train_idx],
                data_A[test_idx], labels[test_idx],
                n_channels=n_channels, n_epochs=n_epochs
            )
            subj_results["A_with_filter"] = metrics_A
            all_A.append(metrics_A["accuracy"])
            print(f"    Accuracy: {metrics_A['accuracy']:.2f}%")
            
            # ----- Condition B: Without Filter -----
            print("\n[B] Without Filter (raw normalized)...")
            data_B = normalize_data(data.copy())
            data_B = resample_data(data_B, target_len=1000)
            
            metrics_B = train_and_eval_ctnet(
                data_B[train_idx], labels[train_idx],
                data_B[test_idx], labels[test_idx],
                n_channels=n_channels, n_epochs=n_epochs
            )
            subj_results["B_without_filter"] = metrics_B
            all_B.append(metrics_B["accuracy"])
            print(f"    Accuracy: {metrics_B['accuracy']:.2f}%")
            
            # Comparison
            diff = metrics_A["accuracy"] - metrics_B["accuracy"]
            subj_results["filter_advantage"] = diff
            results["subjects"][f"S{subj:03d}"] = subj_results
            
            print(f"\n  Filter advantage: {diff:+.2f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["subjects"][f"S{subj:03d}"] = {"error": str(e)}
    
    # Overall summary
    if all_A and all_B:
        filter_advantage = np.mean(all_A) - np.mean(all_B)
        results["summary"] = {
            "with_filter": {
                "mean": float(np.mean(all_A)),
                "std": float(np.std(all_A)),
                "values": all_A
            },
            "without_filter": {
                "mean": float(np.mean(all_B)),
                "std": float(np.std(all_B)),
                "values": all_B
            },
            "filter_advantage": float(filter_advantage),
            "filter_advantage_percentage": float(filter_advantage)
        }
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"With 8-30Hz Filter:    {np.mean(all_A):.2f}% ± {np.std(all_A):.2f}%")
        print(f"Without Filter:        {np.mean(all_B):.2f}% ± {np.std(all_B):.2f}%")
        print()
        print(f"Filter Advantage:      {filter_advantage:+.2f}%")
        print()
        
        if filter_advantage > 0:
            verdict = f"8-30Hz bandpass filter IMPROVES accuracy by {filter_advantage:.2f}%"
            results["summary"]["conclusion"] = "Filter is beneficial"
        else:
            verdict = f"Filter does not improve accuracy (difference: {filter_advantage:.2f}%)"
            results["summary"]["conclusion"] = "Filter shows no benefit in this setup"
        
        print(f"VERDICT: {verdict}")
        results["summary"]["verdict"] = verdict
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Filter Ablation with Full CTNet")
    parser.add_argument("--subjects", nargs="+", type=int, default=[3, 7, 9],
                        help="Subject IDs to test")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Training epochs per condition")
    args = parser.parse_args()
    
    print("="*60)
    print("FILTER ABLATION STUDY")
    print("Using FULL EEGTransformer Model")
    print("="*60)
    print(f"Subjects: {args.subjects}")
    print(f"Epochs: {args.epochs}")
    print()
    print("This experiment will definitively show whether")
    print("8-30Hz bandpass filtering improves classification.")
    
    results = run_experiment(args.subjects, n_epochs=args.epochs)
    
    output_file = OUTPUT_DIR / "filter_ablation_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
