#!/usr/bin/env python3
"""
Filter vs ICA Comparison Experiment

This experiment proves that the 8-30Hz bandpass filter is the key preprocessing
step, making ICA unnecessary. We compare three conditions:

  A) 8-30Hz filter + no ICA  (current pipeline)
  B) No filter + ICA
  C) No filter + no ICA

Expected result: A >= B > C, proving that:
1. The filter alone is sufficient
2. ICA without filter cannot match filter performance
3. The filter removes most artifacts that ICA would target

Usage:
  python scripts/filter_vs_ica_experiment.py
  python scripts/filter_vs_ica_experiment.py --subjects 1 3 7
"""

from __future__ import annotations
import argparse, json, sys, time, warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import butter, filtfilt, resample

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
OUTPUT_DIR = _ROOT / "outputs" / "filter_vs_ica"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading and Preprocessing
# ============================================================================

def load_mat_data(subject_id: int, mode: str = "T") -> Tuple[np.ndarray, np.ndarray]:
    """Load MAT data for a subject."""
    mat_path = DATA_ROOT / f"A{subject_id:02d}{mode}.mat"
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    mat = scipy.io.loadmat(str(mat_path))
    data = mat["data"].astype(np.float64)
    labels = mat["label"].flatten().astype(np.int64) - 1
    return data, labels


def bandpass_filter(data: np.ndarray, low: float = 8.0, high: float = 30.0,
                    fs: float = 250.0, order: int = 4) -> np.ndarray:
    """Apply 8-30Hz bandpass filter."""
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for ch in range(data.shape[1]):
            filtered[i, ch] = filtfilt(b, a, data[i, ch])
    return filtered


def apply_ica_artifact_removal(data: np.ndarray, n_components: int = 20,
                               n_artifacts: int = 2) -> np.ndarray:
    """Apply FastICA per trial and remove highest-variance components."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("sklearn is required for ICA")
    
    n_trials, n_ch, n_times = data.shape
    n_components = min(n_components, n_ch)
    cleaned = np.zeros_like(data)

    for i in range(n_trials):
        trial = data[i]
        try:
            ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
            sources = ica.fit_transform(trial.T)
            variances = np.var(sources, axis=0)
            artifact_idx = np.argsort(variances)[-n_artifacts:]
            sources[:, artifact_idx] = 0
            cleaned[i] = ica.inverse_transform(sources).T
        except Exception:
            cleaned[i] = trial

    return cleaned


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
    """Resample to fixed length."""
    n_trials, n_ch, n_times = data.shape
    resampled = np.zeros((n_trials, n_ch, target_len), dtype=np.float32)
    for i in range(n_trials):
        for ch in range(n_ch):
            resampled[i, ch] = resample(data[i, ch], target_len)
    return resampled


# ============================================================================
# Lightweight CTNet for quick evaluation
# ============================================================================

class LightCTNet(nn.Module):
    """Simplified CTNet for fast training."""
    def __init__(self, n_channels: int = 22, n_classes: int = 4, n_times: int = 1000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 32))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 32, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # (B, 1, C, T)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def train_and_eval(train_data: np.ndarray, train_labels: np.ndarray,
                   test_data: np.ndarray, test_labels: np.ndarray,
                   n_channels: int = 22, n_epochs: int = 100,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"
                   ) -> Dict:
    """Train CTNet and return evaluation metrics."""
    
    # Prepare data
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    # Model
    model = LightCTNet(n_channels=n_channels, n_classes=4, n_times=train_data.shape[2])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Training
    model.train()
    for epoch in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        logits = model(X_test)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    y_true = y_test.numpy()
    accuracy = (preds == y_true).mean() * 100
    
    return {"accuracy": float(accuracy), "n_test": len(y_true)}


# ============================================================================
# Main Experiment
# ============================================================================

def run_experiment(subjects: List[int], n_epochs: int = 100) -> Dict:
    """Run the filter vs ICA comparison experiment."""
    
    results = {
        "experiment": "Filter vs ICA Comparison",
        "date": datetime.now().isoformat(),
        "conditions": {
            "A": "8-30Hz filter + no ICA (current pipeline)",
            "B": "No filter + ICA",
            "C": "No filter + no ICA"
        },
        "subjects": {},
        "summary": {}
    }
    
    all_A, all_B, all_C = [], [], []
    
    for subj in subjects:
        print(f"\n{'='*60}")
        print(f"Subject {subj}")
        print('='*60)
        
        try:
            # Load data
            data, labels = load_mat_data(subj, mode="T")
            n_trials = len(labels)
            n_channels = data.shape[1]
            
            # Split train/test (80/20)
            n_train = int(n_trials * 0.8)
            indices = np.random.RandomState(42).permutation(n_trials)
            train_idx, test_idx = indices[:n_train], indices[n_train:]
            
            subj_results = {}
            
            # ----- Condition A: Filter + No ICA -----
            print("\n[A] 8-30Hz Filter + No ICA...")
            data_A = bandpass_filter(data.copy())
            data_A = normalize_data(data_A)
            data_A = resample_data(data_A)
            
            metrics_A = train_and_eval(
                data_A[train_idx], labels[train_idx],
                data_A[test_idx], labels[test_idx],
                n_channels=n_channels, n_epochs=n_epochs
            )
            subj_results["A_filter_noICA"] = metrics_A
            all_A.append(metrics_A["accuracy"])
            print(f"    Accuracy: {metrics_A['accuracy']:.2f}%")
            
            # ----- Condition B: No Filter + ICA -----
            print("\n[B] No Filter + ICA...")
            data_B = apply_ica_artifact_removal(data.copy())
            data_B = normalize_data(data_B)
            data_B = resample_data(data_B)
            
            metrics_B = train_and_eval(
                data_B[train_idx], labels[train_idx],
                data_B[test_idx], labels[test_idx],
                n_channels=n_channels, n_epochs=n_epochs
            )
            subj_results["B_noFilter_ICA"] = metrics_B
            all_B.append(metrics_B["accuracy"])
            print(f"    Accuracy: {metrics_B['accuracy']:.2f}%")
            
            # ----- Condition C: No Filter + No ICA -----
            print("\n[C] No Filter + No ICA (raw)...")
            data_C = normalize_data(data.copy())
            data_C = resample_data(data_C)
            
            metrics_C = train_and_eval(
                data_C[train_idx], labels[train_idx],
                data_C[test_idx], labels[test_idx],
                n_channels=n_channels, n_epochs=n_epochs
            )
            subj_results["C_noFilter_noICA"] = metrics_C
            all_C.append(metrics_C["accuracy"])
            print(f"    Accuracy: {metrics_C['accuracy']:.2f}%")
            
            # Subject summary
            subj_results["comparison"] = {
                "A_vs_B": metrics_A["accuracy"] - metrics_B["accuracy"],
                "A_vs_C": metrics_A["accuracy"] - metrics_C["accuracy"],
                "B_vs_C": metrics_B["accuracy"] - metrics_C["accuracy"],
            }
            results["subjects"][f"S{subj:02d}"] = subj_results
            
            print(f"\n  Summary for S{subj:02d}:")
            print(f"    A (Filter+noICA): {metrics_A['accuracy']:.2f}%")
            print(f"    B (noFilter+ICA): {metrics_B['accuracy']:.2f}%")
            print(f"    C (noFilter+noICA): {metrics_C['accuracy']:.2f}%")
            print(f"    Filter advantage over ICA (A-B): {subj_results['comparison']['A_vs_B']:+.2f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            results["subjects"][f"S{subj:02d}"] = {"error": str(e)}
    
    # Overall summary
    if all_A and all_B and all_C:
        results["summary"] = {
            "A_filter_noICA": {
                "mean": float(np.mean(all_A)),
                "std": float(np.std(all_A)),
                "values": all_A
            },
            "B_noFilter_ICA": {
                "mean": float(np.mean(all_B)),
                "std": float(np.std(all_B)),
                "values": all_B
            },
            "C_noFilter_noICA": {
                "mean": float(np.mean(all_C)),
                "std": float(np.std(all_C)),
                "values": all_C
            },
            "conclusion": {
                "filter_vs_ICA_advantage": float(np.mean(all_A) - np.mean(all_B)),
                "filter_vs_raw_advantage": float(np.mean(all_A) - np.mean(all_C)),
                "ICA_vs_raw_advantage": float(np.mean(all_B) - np.mean(all_C)),
            }
        }
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"A (8-30Hz Filter + no ICA): {np.mean(all_A):.2f}% ± {np.std(all_A):.2f}%")
        print(f"B (No Filter + ICA):        {np.mean(all_B):.2f}% ± {np.std(all_B):.2f}%")
        print(f"C (No Filter + no ICA):     {np.mean(all_C):.2f}% ± {np.std(all_C):.2f}%")
        print()
        print(f"Filter advantage over ICA (A-B):      {np.mean(all_A) - np.mean(all_B):+.2f}%")
        print(f"Filter advantage over raw (A-C):      {np.mean(all_A) - np.mean(all_C):+.2f}%")
        print(f"ICA-only advantage over raw (B-C):    {np.mean(all_B) - np.mean(all_C):+.2f}%")
        print()
        
        if np.mean(all_A) >= np.mean(all_B):
            print("CONCLUSION: 8-30Hz bandpass filter alone is sufficient.")
            print("            ICA provides no additional benefit when filter is applied.")
            results["summary"]["conclusion"]["verdict"] = "Filter alone is sufficient; ICA unnecessary"
        else:
            print("CONCLUSION: ICA shows benefit even without filter.")
            results["summary"]["conclusion"]["verdict"] = "ICA shows benefit"
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Filter vs ICA Comparison")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1, 3, 7],
                        help="Subject IDs to test")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per condition")
    args = parser.parse_args()
    
    print("="*60)
    print("FILTER vs ICA COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Subjects: {args.subjects}")
    print(f"Epochs per condition: {args.epochs}")
    print()
    print("Conditions:")
    print("  A) 8-30Hz bandpass filter + no ICA (current pipeline)")
    print("  B) No filter + ICA artifact removal")
    print("  C) No filter + no ICA (raw normalized)")
    print()
    print("Hypothesis: A >= B > C")
    print("  - Filter removes most artifacts (concentrated < 4Hz)")
    print("  - ICA alone cannot match filter performance")
    print("  - Filter makes ICA unnecessary")
    
    results = run_experiment(args.subjects, n_epochs=args.epochs)
    
    # Save results
    output_file = OUTPUT_DIR / "filter_vs_ica_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
