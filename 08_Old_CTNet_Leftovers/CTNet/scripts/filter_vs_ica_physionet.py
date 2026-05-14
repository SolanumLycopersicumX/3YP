#!/usr/bin/env python3
"""
Filter vs ICA Comparison on PhysioNet Raw EEG Data

This experiment uses truly raw EEG from PhysioNet to prove that the 8-30Hz 
bandpass filter is essential, making ICA unnecessary.

Conditions:
  A) 8-30Hz filter + no ICA  (current pipeline)
  B) No filter + ICA
  C) No filter + no ICA (truly raw)

Usage:
  python scripts/filter_vs_ica_physionet.py --subjects 3 7 9
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

try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("WARNING: MNE required for loading PhysioNet data")

try:
    from sklearn.decomposition import FastICA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

PHYSIONET_DIR = _ROOT / "physionet_raw"
OUTPUT_DIR = _ROOT / "outputs" / "filter_vs_ica_physionet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PhysioNet Data Loading
# ============================================================================

def load_physionet_subject(subject_id: int, task_runs: List[int] = [4, 8, 12]) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw EEG from PhysioNet for a subject."""
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
        
        # Get events
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        
        # Create epochs (no filtering here - raw data)
        epochs = mne.Epochs(raw, events, event_id, tmin=0, tmax=4.0,
                           baseline=None, preload=True, verbose=False)
        
        data = epochs.get_data()  # (n_epochs, n_channels, n_times)
        labels = epochs.events[:, 2]
        
        all_epochs.append(data)
        all_labels.append(labels)
    
    if not all_epochs:
        raise ValueError(f"No data found for subject {subject_id}")
    
    X = np.concatenate(all_epochs, axis=0)
    y = np.concatenate(all_labels, axis=0)
    
    # Map to 0-indexed labels
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


def apply_ica_artifact_removal(data: np.ndarray, n_components: int = 20,
                               n_artifacts: int = 2) -> np.ndarray:
    """Apply FastICA per trial."""
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


def resample_data(data: np.ndarray, target_len: int = 640) -> np.ndarray:
    """Resample to fixed length."""
    n_trials, n_ch, n_times = data.shape
    resampled = np.zeros((n_trials, n_ch, target_len), dtype=np.float32)
    for i in range(n_trials):
        for ch in range(n_ch):
            resampled[i, ch] = resample(data[i, ch], target_len)
    return resampled


# ============================================================================
# Model
# ============================================================================

class LightCTNet(nn.Module):
    """Simplified CTNet."""
    def __init__(self, n_channels: int = 64, n_classes: int = 4, n_times: int = 640):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 64), padding=(0, 32))
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1))
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 32))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32 * 32, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def train_and_eval(train_data: np.ndarray, train_labels: np.ndarray,
                   test_data: np.ndarray, test_labels: np.ndarray,
                   n_channels: int = 64, n_epochs: int = 100,
                   device: str = "cuda" if torch.cuda.is_available() else "cpu"
                   ) -> Dict:
    """Train and evaluate."""
    X_train = torch.tensor(train_data, dtype=torch.float32)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor(test_data, dtype=torch.float32)
    y_test = torch.tensor(test_labels, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    
    n_classes = len(np.unique(train_labels))
    model = LightCTNet(n_channels=n_channels, n_classes=n_classes, n_times=train_data.shape[2])
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
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
    """Run filter vs ICA comparison on PhysioNet raw data."""
    
    results = {
        "experiment": "Filter vs ICA on PhysioNet Raw EEG",
        "date": datetime.now().isoformat(),
        "data_source": "PhysioNet EEG Motor Movement/Imagery Database (raw EDF)",
        "conditions": {
            "A": "8-30Hz bandpass filter + no ICA",
            "B": "No filter + ICA artifact removal",
            "C": "No filter + no ICA (raw normalized)"
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
            # Load raw data
            print("Loading raw EEG data...")
            data, labels = load_physionet_subject(subj)
            n_trials = len(labels)
            n_channels = data.shape[1]
            
            print(f"  Data shape: {data.shape}")
            print(f"  Raw data range: [{data.min():.2f}, {data.max():.2f}] (microvolts)")
            
            # Split train/test
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
            data_B = apply_ica_artifact_removal(data.copy(), n_components=min(20, n_channels))
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
            results["subjects"][f"S{subj:03d}"] = subj_results
            
            print(f"\n  Summary for S{subj:03d}:")
            print(f"    A (Filter+noICA): {metrics_A['accuracy']:.2f}%")
            print(f"    B (noFilter+ICA): {metrics_B['accuracy']:.2f}%")
            print(f"    C (noFilter+noICA): {metrics_C['accuracy']:.2f}%")
            print(f"    Filter advantage over ICA (A-B): {subj_results['comparison']['A_vs_B']:+.2f}%")
            print(f"    Filter advantage over raw (A-C): {subj_results['comparison']['A_vs_C']:+.2f}%")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results["subjects"][f"S{subj:03d}"] = {"error": str(e)}
    
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
                "filter_vs_ICA": float(np.mean(all_A) - np.mean(all_B)),
                "filter_vs_raw": float(np.mean(all_A) - np.mean(all_C)),
                "ICA_vs_raw": float(np.mean(all_B) - np.mean(all_C)),
            }
        }
        
        print("\n" + "="*60)
        print("OVERALL SUMMARY")
        print("="*60)
        print(f"A (8-30Hz Filter + no ICA): {np.mean(all_A):.2f}% ± {np.std(all_A):.2f}%")
        print(f"B (No Filter + ICA):        {np.mean(all_B):.2f}% ± {np.std(all_B):.2f}%")
        print(f"C (No Filter + no ICA):     {np.mean(all_C):.2f}% ± {np.std(all_C):.2f}%")
        print()
        print(f"Filter advantage over ICA (A-B):   {np.mean(all_A) - np.mean(all_B):+.2f}%")
        print(f"Filter advantage over raw (A-C):   {np.mean(all_A) - np.mean(all_C):+.2f}%")
        print(f"ICA-only advantage over raw (B-C): {np.mean(all_B) - np.mean(all_C):+.2f}%")
        print()
        
        # Determine verdict
        filter_better = np.mean(all_A) > np.mean(all_B) and np.mean(all_A) > np.mean(all_C)
        if filter_better:
            verdict = "8-30Hz bandpass filter is essential and makes ICA unnecessary"
        elif np.mean(all_A) > np.mean(all_C):
            verdict = "Filter improves over raw data; ICA comparison is mixed"
        else:
            verdict = "Unexpected results - further investigation needed"
        
        results["summary"]["conclusion"]["verdict"] = verdict
        print(f"VERDICT: {verdict}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Filter vs ICA on PhysioNet")
    parser.add_argument("--subjects", nargs="+", type=int, default=[3, 7, 9],
                        help="Subject IDs to test")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs per condition")
    args = parser.parse_args()
    
    print("="*60)
    print("FILTER vs ICA on PhysioNet Raw EEG")
    print("="*60)
    print(f"Subjects: {args.subjects}")
    print(f"Epochs: {args.epochs}")
    print()
    print("Using truly raw EEG data from PhysioNet EDF files")
    print("(Not pre-processed MAT files)")
    
    results = run_experiment(args.subjects, n_epochs=args.epochs)
    
    output_file = OUTPUT_DIR / "filter_vs_ica_physionet_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
