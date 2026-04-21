#!/usr/bin/env python3
"""
Evaluate PhysioNet-trained CTNet classification performance

Loads a trained model and runs comprehensive evaluation:
  - Per-class accuracy, precision, recall, F1
  - Confusion matrix (normalized + counts)
  - Per-subject breakdown (if multiple subjects)
  - t-SNE / feature visualization
  - Direction mapping analysis (for robot arm control)

Example:
  python scripts/eval_physionet_ctnet.py --model outputs/physionet_ctnet/pool/model_pool.pth --subjects 1 2 3 4 5
  python scripts/eval_physionet_ctnet.py --model outputs/physionet_ctnet/pool/model_pool.pth --subjects 11 12 13 --unseen
"""

from __future__ import annotations

import argparse
import sys
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.signal import resample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from utils import calMetrics
from scripts.physionet_loader import (
    load_subject_4class_mne,
    load_multiple_subjects_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)

TARGET_SAMPLES = 1000
CLASS_NAMES = ['Left', 'Right', 'Hands/Up', 'Feet/Down']
DIRECTION_MAP = {
    0: ('Left',      '←',  (-1,  0)),
    1: ('Right',     '→',  ( 1,  0)),
    2: ('Hands/Up',  '↑',  ( 0,  1)),
    3: ('Feet/Down', '↓',  ( 0, -1)),
}


def resample_epochs(data: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    n_epochs, n_ch, n_times = data.shape
    if n_times == target_len:
        return data
    resampled = np.zeros((n_epochs, n_ch, target_len), dtype=np.float32)
    for i in range(n_epochs):
        resampled[i] = resample(data[i], target_len, axis=1)
    return resampled


def load_model(model_path: Path, device: torch.device):
    """Load trained CTNet model and metadata."""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()

    meta_path = model_path.with_suffix('.json')
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    return model, meta


def load_test_data(
    subjects: List[int],
    meta: dict,
    bandpass: bool = True,
    verbose: bool = True,
):
    """Load and preprocess test data, applying same normalization as training.
    Loads subjects individually to handle differing epoch lengths."""
    all_data, all_labels, all_sids = [], [], []

    for sub in subjects:
        try:
            if verbose:
                print(f"  Loading S{sub:03d}...", end=" ", flush=True)
            d, l = load_subject_4class_mne(sub)
            if bandpass:
                d = preprocess_data(d)
            d = resample_epochs(d, TARGET_SAMPLES)
            all_data.append(d)
            all_labels.append(l)
            all_sids.append(np.full(len(l), sub))
            if verbose:
                print(f"OK ({len(l)} epochs)")
        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_sids)

    norm_mean = meta.get('norm_mean', data.mean())
    norm_std = meta.get('norm_std', data.std())
    data = (data - norm_mean) / (norm_std + 1e-8)

    return data, labels, subject_ids


def classify_all(
    model: nn.Module,
    data: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
):
    """Run classification on all data. Returns predictions and logits."""
    ds = TensorDataset(
        torch.from_numpy(np.expand_dims(data, 1)).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_logits = []
    all_features = []

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            features, logits = model(batch_x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_features.append(features.cpu().numpy())

    return (
        np.concatenate(all_preds),
        np.concatenate(all_logits),
        np.concatenate(all_features),
    )


def evaluate_overall(y_true, y_pred, out_dir: Path, tag: str):
    """Compute overall metrics and save confusion matrix."""
    from sklearn.metrics import confusion_matrix, classification_report

    acc, prec, rec, f1, kappa = calMetrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  Overall Results ({tag})")
    print(f"{'='*60}")
    print(f"  Accuracy:   {acc*100:.2f}%")
    print(f"  Precision:  {prec*100:.2f}%")
    print(f"  Recall:     {rec*100:.2f}%")
    print(f"  F1 Score:   {f1*100:.2f}%")
    print(f"  Kappa:      {kappa:.4f}")
    print(f"  Chance:     25.00%")
    print()

    print("  Per-class breakdown:")
    for i, name in enumerate(CLASS_NAMES):
        cls_data = report.get(name, {})
        p = cls_data.get('precision', 0) * 100
        r = cls_data.get('recall', 0) * 100
        f = cls_data.get('f1-score', 0) * 100
        n = int(cls_data.get('support', 0))
        print(f"    {name:12s}  Prec={p:5.1f}%  Rec={r:5.1f}%  F1={f:5.1f}%  (n={n})")

    metrics = {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'kappa': float(kappa),
        'classification_report': report,
    }

    with open(out_dir / f"eval_metrics_{tag}.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # --- Confusion matrix ---
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    cm_pct = np.nan_to_num(cm_pct)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))

    # Normalized
    im0 = axes[0].imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    fig.colorbar(im0, ax=axes[0], label='%', shrink=0.8)
    for i in range(4):
        for j in range(4):
            c = 'white' if cm_pct[i, j] > 50 else 'black'
            axes[0].text(j, i, f'{cm_pct[i,j]:.1f}%', ha='center', va='center',
                         fontsize=12, color=c, fontweight='bold')
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels(CLASS_NAMES, fontsize=10)
    axes[0].set_yticklabels(CLASS_NAMES, fontsize=10)
    axes[0].set_xlabel('Predicted', fontweight='bold')
    axes[0].set_ylabel('True', fontweight='bold')
    axes[0].set_title('Normalized (%)', fontweight='bold')

    # Counts
    im1 = axes[1].imshow(cm, cmap='Blues')
    fig.colorbar(im1, ax=axes[1], label='Count', shrink=0.8)
    for i in range(4):
        for j in range(4):
            max_val = cm.max()
            c = 'white' if cm[i, j] > max_val * 0.5 else 'black'
            axes[1].text(j, i, f'{cm[i,j]}', ha='center', va='center',
                         fontsize=12, color=c, fontweight='bold')
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels(CLASS_NAMES, fontsize=10)
    axes[1].set_yticklabels(CLASS_NAMES, fontsize=10)
    axes[1].set_xlabel('Predicted', fontweight='bold')
    axes[1].set_ylabel('True', fontweight='bold')
    axes[1].set_title('Raw Counts', fontweight='bold')

    fig.suptitle(f'CTNet PhysioNet — Confusion Matrix ({tag})\n'
                 f'Accuracy: {acc:.2%}  |  Kappa: {kappa:.4f}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()

    return metrics


def evaluate_per_subject(
    y_true, y_pred, subject_ids, subjects, out_dir: Path, tag: str
):
    """Per-subject accuracy breakdown."""
    results = []

    for sub in subjects:
        mask = subject_ids == sub
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        acc = (yt == yp).mean()
        results.append({'subject': int(sub), 'accuracy': float(acc), 'n_trials': int(mask.sum())})

    if not results:
        return results

    accs = [r['accuracy'] for r in results]
    subs = [r['subject'] for r in results]

    print(f"\n  Per-subject accuracy:")
    for r in results:
        bar = '█' * int(r['accuracy'] * 40)
        print(f"    S{r['subject']:03d}: {r['accuracy']*100:5.1f}% ({r['n_trials']:3d} trials) {bar}")
    print(f"    {'Mean':>4s}: {np.mean(accs)*100:5.1f}% ± {np.std(accs)*100:.1f}%")

    fig, ax = plt.subplots(figsize=(max(8, len(subs)*0.6), 5))
    colors = ['#2ecc71' if a > 0.5 else '#e74c3c' if a < 0.3 else '#f39c12' for a in accs]
    bars = ax.bar(range(len(subs)), [a*100 for a in accs], color=colors, alpha=0.85, edgecolor='white')

    ax.axhline(y=np.mean(accs)*100, color='blue', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(accs)*100:.1f}%', alpha=0.8)
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Chance (25%)')

    for bar, acc_val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc_val*100:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([f'S{s:03d}' for s in subs], rotation=45, fontsize=9)
    ax.set_xlabel('Subject')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'CTNet PhysioNet — Per-Subject Accuracy ({tag})', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"per_subject_accuracy_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()

    with open(out_dir / f"per_subject_{tag}.json", 'w') as f:
        json.dump({
            'results': results,
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
        }, f, indent=2)

    return results


def plot_direction_mapping(y_true, y_pred, out_dir: Path, tag: str):
    """Visualize how classification maps to robot arm directions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: True vs Predicted direction distribution
    for cls_id, (name, arrow, _) in DIRECTION_MAP.items():
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        preds = y_pred[mask]
        pred_counts = np.bincount(preds, minlength=4)
        total = mask.sum()

        x = np.arange(4)
        colors_bar = ['#3498db' if i == cls_id else '#e74c3c' for i in range(4)]
        offset = cls_id * 0.2 - 0.3
        axes[0].bar(x + offset, pred_counts / total * 100, width=0.18,
                    label=f'True={name}', alpha=0.8)

    axes[0].set_xticks(range(4))
    axes[0].set_xticklabels([f'{DIRECTION_MAP[i][1]} {DIRECTION_MAP[i][0]}' for i in range(4)])
    axes[0].set_ylabel('Predicted (%)')
    axes[0].set_title('Prediction Distribution by True Class', fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis='y', alpha=0.3)

    # Right: Direction accuracy compass
    ax2 = axes[1]
    correct = (y_true == y_pred)

    for cls_id, (name, arrow, (dx, dy)) in DIRECTION_MAP.items():
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        cls_acc = correct[mask].mean()
        color = '#2ecc71' if cls_acc > 0.4 else '#e74c3c' if cls_acc < 0.25 else '#f39c12'
        ax2.annotate('', xy=(dx * 0.8, dy * 0.8), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color=color, lw=3))
        ax2.text(dx * 1.05, dy * 1.05, f'{arrow} {name}\n{cls_acc*100:.1f}%',
                 ha='center', va='center', fontsize=11, fontweight='bold', color=color)

    overall_acc = correct.mean()
    ax2.text(0, 0, f'Overall\n{overall_acc*100:.1f}%', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Direction Classification Accuracy', fontweight='bold')
    ax2.axis('off')

    fig.suptitle(f'CTNet PhysioNet — Direction Mapping Analysis ({tag})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"direction_mapping_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()


def plot_confidence_distribution(logits, y_true, y_pred, out_dir: Path, tag: str):
    """Plot classification confidence (softmax probability) distribution."""
    probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    max_probs = probs.max(axis=1)

    correct_mask = y_true == y_pred
    correct_conf = max_probs[correct_mask]
    wrong_conf = max_probs[~correct_mask]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confidence histogram
    bins = np.linspace(0, 1, 30)
    if len(correct_conf) > 0:
        axes[0].hist(correct_conf, bins=bins, alpha=0.7, color='#2ecc71', label=f'Correct (n={len(correct_conf)})')
    if len(wrong_conf) > 0:
        axes[0].hist(wrong_conf, bins=bins, alpha=0.7, color='#e74c3c', label=f'Wrong (n={len(wrong_conf)})')
    axes[0].axvline(x=0.25, color='gray', linestyle=':', alpha=0.5, label='Chance (0.25)')
    axes[0].set_xlabel('Max Softmax Probability')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Classification Confidence', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-class confidence
    for cls_id in range(4):
        mask = y_true == cls_id
        if mask.sum() == 0:
            continue
        cls_conf = max_probs[mask]
        axes[1].boxplot(cls_conf, positions=[cls_id], widths=0.6)

    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(CLASS_NAMES)
    axes[1].set_ylabel('Max Softmax Probability')
    axes[1].set_title('Confidence by True Class', fontweight='bold')
    axes[1].axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    axes[1].grid(True, axis='y', alpha=0.3)

    fig.suptitle(f'CTNet PhysioNet — Confidence Analysis ({tag})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"confidence_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()


def generate_report(metrics, per_subject_results, out_dir: Path, tag: str, args):
    """Generate a text summary report."""
    lines = [
        "=" * 60,
        "  CTNet PhysioNet Evaluation Report",
        "=" * 60,
        f"  Model:    {args.model}",
        f"  Subjects: {args.subjects}",
        f"  Unseen:   {args.unseen}",
        f"  Tag:      {tag}",
        "",
        "  Overall Metrics:",
        f"    Accuracy:   {metrics['accuracy']*100:.2f}%",
        f"    Precision:  {metrics['precision']*100:.2f}%",
        f"    Recall:     {metrics['recall']*100:.2f}%",
        f"    F1 Score:   {metrics['f1']*100:.2f}%",
        f"    Kappa:      {metrics['kappa']:.4f}",
        "",
    ]

    if per_subject_results:
        accs = [r['accuracy'] for r in per_subject_results]
        lines.append("  Per-Subject Results:")
        for r in per_subject_results:
            lines.append(f"    S{r['subject']:03d}: {r['accuracy']*100:5.1f}% ({r['n_trials']} trials)")
        lines.append(f"    Mean:  {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
        lines.append("")

    lines.append("  Direction Mapping (for robot arm control):")
    lines.append("    Class 0 (Left)       → Move Left  (dx=-1)")
    lines.append("    Class 1 (Right)      → Move Right (dx=+1)")
    lines.append("    Class 2 (Hands/Up)   → Move Up    (dy=+1)")
    lines.append("    Class 3 (Feet/Down)  → Move Down  (dy=-1)")
    lines.append("")
    lines.append("=" * 60)

    report_text = "\n".join(lines)
    print(report_text)

    with open(out_dir / f"report_{tag}.txt", 'w') as f:
        f.write(report_text)


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate PhysioNet-trained CTNet")

    p.add_argument("--model", type=Path, required=True,
                   help="Path to trained model (.pth)")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                   help="Test subject IDs")
    p.add_argument("--unseen", action="store_true",
                   help="Flag: subjects were NOT in training set")
    p.add_argument("--filter", action="store_true", default=True,
                   help="Apply 8-30 Hz bandpass (default: True)")
    p.add_argument("--no-filter", dest="filter", action="store_false")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: model's parent/eval/)")
    p.add_argument("--tag", type=str, default=None,
                   help="Tag for output files")

    return p.parse_args()


def main():
    args = parse_args()

    if not MNE_AVAILABLE:
        print("MNE is required: pip install mne")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  CTNet PhysioNet Evaluation")
    print("=" * 60)
    print(f"  Model:    {args.model}")
    print(f"  Subjects: {args.subjects}")
    print(f"  Unseen:   {args.unseen}")
    print(f"  Device:   {device}")

    model, meta = load_model(args.model, device)
    print(f"  Meta:     {meta.get('dataset', 'unknown')}")
    print(f"  Trained:  subjects={meta.get('subjects', '?')}")

    if args.output_dir is None:
        args.output_dir = args.model.parent / "eval"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag or ("unseen" if args.unseen else "test")

    print(f"\nLoading test data...")
    data, labels, subject_ids = load_test_data(
        args.subjects, meta, bandpass=args.filter, verbose=True
    )

    print(f"\nRunning classification...")
    y_pred, logits, features = classify_all(model, data, device, args.batch_size)

    metrics = evaluate_overall(labels, y_pred, args.output_dir, tag)
    per_sub = evaluate_per_subject(labels, y_pred, subject_ids, args.subjects, args.output_dir, tag)
    plot_direction_mapping(labels, y_pred, args.output_dir, tag)
    plot_confidence_distribution(logits, labels, y_pred, args.output_dir, tag)
    generate_report(metrics, per_sub, args.output_dir, tag, args)

    print(f"\nAll outputs saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
