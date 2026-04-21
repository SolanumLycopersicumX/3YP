#!/usr/bin/env python3
"""
Filter Ablation Study via Fine-tuning

This script performs subject-specific fine-tuning with and without 8-30Hz filtering
to definitively demonstrate the filter's contribution to model accuracy.

We use the exact same methodology as the original fine-tuning (88.78% result):
- Same pre-trained model
- Same subjects
- Same 5-fold CV
- Only difference: whether filtering is applied
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import resample

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from CTNet_model import EEGTransformer
from scripts.physionet_loader import (
    load_subject_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)

TARGET_SAMPLES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resample_epochs(data: np.ndarray, target: int = TARGET_SAMPLES) -> np.ndarray:
    """Resample each epoch to target length."""
    n_epochs, n_ch, n_time = data.shape
    resampled = np.zeros((n_epochs, n_ch, target), dtype=np.float32)
    for i in range(n_epochs):
        for c in range(n_ch):
            resampled[i, c, :] = resample(data[i, c, :], target)
    return resampled


def load_subject_data(subject_id: int, apply_filter: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Load and optionally filter subject data."""
    try:
        data, labels = load_subject_4class_mne(subject_id)
    except Exception as e:
        print(f"  Failed to load subject {subject_id}: {e}")
        return None, None
    
    if apply_filter:
        data = preprocess_data(data)  # 8-30 Hz bandpass
    
    data = resample_epochs(data, TARGET_SAMPLES)
    return data, labels


def finetune_fold(
    pretrained_path: Path,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_channels: int = 64,
    n_epochs: int = 50,
    lr: float = 1e-4,
) -> float:
    """Fine-tune model on one fold and return test accuracy."""
    
    # Load pre-trained model
    loaded = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
    
    # Handle both full model and state_dict saves
    if isinstance(loaded, nn.Module):
        model = loaded
    else:
        model = EEGTransformer(
            heads=2, depth=6, emb_size=40,
            database_type='P',  # PhysioNet
            eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
            eeg1_pooling_size1=8, eeg1_pooling_size2=8,
            eeg1_dropout_rate=0.3, eeg1_number_channel=n_channels,
            flatten_eeg1=600,
        )
        model.load_state_dict(loaded, strict=False)
    
    model = model.to(DEVICE)
    
    # Prepare data
    X_tr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_te = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Fine-tune
    model.train()
    for _ in range(n_epochs):
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            _, out = model(X_batch)  # model returns (features, logits)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        _, logits = model(X_te.to(DEVICE))  # model returns (features, logits)
        preds = logits.argmax(dim=1).cpu().numpy()
    
    accuracy = (preds == y_te.numpy()).mean()
    return accuracy


def run_subject_cv(
    subject_id: int,
    pretrained_path: Path,
    apply_filter: bool,
    n_folds: int = 5,
) -> Dict[str, Any]:
    """Run 5-fold CV fine-tuning for one subject."""
    
    data, labels = load_subject_data(subject_id, apply_filter=apply_filter)
    if data is None:
        return None
    
    n_channels = data.shape[1]
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_accs = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(data, labels), 1):
        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]
        
        acc = finetune_fold(
            pretrained_path,
            X_train, y_train,
            X_test, y_test,
            n_channels=n_channels,
        )
        fold_accs.append(acc)
    
    return {
        "subject": subject_id,
        "accuracy": float(np.mean(fold_accs)),
        "std": float(np.std(fold_accs)),
        "folds": fold_accs,
    }


def main():
    parser = argparse.ArgumentParser(description="Filter ablation via fine-tuning")
    parser.add_argument("--subjects", nargs="+", type=int,
                        default=[3, 7, 9, 38, 43],
                        help="Subject IDs to test")
    parser.add_argument("--pretrained", type=Path,
                        default=ROOT / "outputs/physionet_ctnet_109sub_v2/pool/model_pool.pth",
                        help="Path to pre-trained model")
    args = parser.parse_args()
    
    output_dir = ROOT / "outputs" / "filter_ablation_finetune"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("FILTER ABLATION STUDY VIA FINE-TUNING")
    print("="*60)
    print(f"Pre-trained model: {args.pretrained}")
    print(f"Subjects: {args.subjects}")
    print(f"Device: {DEVICE}")
    print()
    
    results = {
        "with_filter": [],
        "without_filter": [],
    }
    
    for subject in args.subjects:
        print(f"\n{'='*50}")
        print(f"Subject {subject}")
        print('='*50)
        
        # WITH filter
        print("\n[A] WITH 8-30Hz Filter...")
        result_A = run_subject_cv(subject, args.pretrained, apply_filter=True)
        if result_A:
            results["with_filter"].append(result_A)
            print(f"    Accuracy: {result_A['accuracy']*100:.2f}% ± {result_A['std']*100:.2f}%")
        
        # WITHOUT filter
        print("\n[B] WITHOUT Filter...")
        result_B = run_subject_cv(subject, args.pretrained, apply_filter=False)
        if result_B:
            results["without_filter"].append(result_B)
            print(f"    Accuracy: {result_B['accuracy']*100:.2f}% ± {result_B['std']*100:.2f}%")
        
        if result_A and result_B:
            diff = (result_A['accuracy'] - result_B['accuracy']) * 100
            print(f"\n    Filter advantage: {diff:+.2f}%")
    
    # Summary
    print("\n" + "="*60)
    print("OVERALL SUMMARY")
    print("="*60)
    
    if results["with_filter"] and results["without_filter"]:
        mean_with = np.mean([r["accuracy"] for r in results["with_filter"]])
        mean_without = np.mean([r["accuracy"] for r in results["without_filter"]])
        
        print(f"WITH 8-30Hz Filter:    {mean_with*100:.2f}%")
        print(f"WITHOUT Filter:        {mean_without*100:.2f}%")
        print()
        
        advantage = (mean_with - mean_without) * 100
        print(f"FILTER ADVANTAGE:      {advantage:+.2f}%")
        print()
        
        if advantage > 0:
            conclusion = f"8-30Hz bandpass filter IMPROVES fine-tuning accuracy by {advantage:.2f}%"
        elif advantage < -2:
            conclusion = f"Filter appears to HURT performance (difference: {advantage:.2f}%)"
        else:
            conclusion = "Filter shows minimal effect on fine-tuning accuracy"
        
        print(f"CONCLUSION: {conclusion}")
        
        results["summary"] = {
            "mean_with_filter": float(mean_with),
            "mean_without_filter": float(mean_without),
            "filter_advantage": float(advantage),
            "conclusion": conclusion,
        }
    
    # Save results
    output_file = output_dir / "filter_ablation_finetune_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
