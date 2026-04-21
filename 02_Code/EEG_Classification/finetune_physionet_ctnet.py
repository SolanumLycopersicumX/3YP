#!/usr/bin/env python3
"""
Fine-tune pretrained CTNet on specific PhysioNet subjects.

Strategy: Load 109-subject pretrained model, freeze CNN backbone,
fine-tune transformer + classifier on target subject(s).
This avoids overfitting while adapting to individual EEG patterns.

Example:
  python scripts/finetune_physionet_ctnet.py \
    --pretrained outputs/physionet_ctnet_109sub_v2/pool/model_pool.pth \
    --subjects 7 15 48 \
    --epochs 200
"""

from __future__ import annotations

import argparse
import sys
import json
import time
import datetime
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from scipy.signal import resample

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from CTNet_model import EEGTransformer
from utils import calMetrics
from scripts.physionet_loader import (
    load_subject_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)

TARGET_SAMPLES = 1000
N_CLASSES = 4
CLASS_NAMES = ['Left', 'Right', 'Hands/Up', 'Feet/Down']


def resample_epochs(data, target_len=TARGET_SAMPLES):
    n, c, t = data.shape
    if t == target_len:
        return data
    out = np.zeros((n, c, target_len), dtype=np.float32)
    for i in range(n):
        out[i] = resample(data[i], target_len, axis=1)
    return out


def interaug(data, labels, n_classes=4, n_aug=4, n_seg=8, batch_size=32):
    seg_len = data.shape[2] // n_seg
    aug_d, aug_l = [], []
    desired = max(1, n_aug * max(1, batch_size // n_classes))
    for cls in range(n_classes):
        mask = labels == cls
        cd = data[mask]
        if len(cd) == 0:
            continue
        nr = min(desired, len(cd))
        tmp = np.zeros((nr, data.shape[1], data.shape[2]), dtype=np.float32)
        for ri in range(nr):
            for rj in range(n_seg):
                src = np.random.randint(0, len(cd))
                s = rj * seg_len
                tmp[ri, :, s:s+seg_len] = cd[src, :, s:s+seg_len]
        aug_d.append(tmp)
        aug_l.append(np.full(nr, cls, dtype=np.int64))
    ad = np.concatenate(aug_d)
    al = np.concatenate(aug_l)
    idx = np.random.permutation(len(ad))
    return ad[idx], al[idx]


def finetune_subject(
    model: EEGTransformer,
    subject: int,
    args,
    device: torch.device,
    norm_mean: float,
    norm_std: float,
    channel_indices: list = None,
) -> dict:
    """Fine-tune on one subject using 5-fold cross-validation."""

    print(f"\n{'='*60}")
    print(f"  Fine-tuning on Subject {subject}")
    print(f"{'='*60}")

    try:
        data, labels = load_subject_4class_mne(subject)
    except Exception as e:
        print(f"  Failed: {e}")
        return None

    data = preprocess_data(data)
    data = resample_epochs(data, TARGET_SAMPLES)
    if channel_indices is not None:
        data = data[:, channel_indices, :]
    data = (data - norm_mean) / (norm_std + 1e-8)

    print(f"  Data: {data.shape}, Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accs = []
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(data, labels)):
        X_train, y_train = data[train_idx], labels[train_idx]
        X_test, y_test = data[test_idx], labels[test_idx]

        ft_model = _create_finetune_model(model, args, device)

        test_ds = TensorDataset(
            torch.from_numpy(np.expand_dims(X_test, 1)).float(),
            torch.from_numpy(y_test).long(),
        )
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

        best_model, best_acc = _finetune_loop(
            ft_model, X_train, y_train, test_loader, device, args,
            label=f"S{subject:03d}-F{fold_i+1}"
        )

        # Final eval
        best_model.eval()
        preds = []
        with torch.no_grad():
            for (bx, _) in test_loader:
                _, logits = best_model(bx.to(device))
                preds.append(logits.argmax(1).cpu().numpy())
        y_pred = np.concatenate(preds)
        acc = (y_test == y_pred).mean()
        fold_accs.append(acc)

        print(f"    Fold {fold_i+1}: {acc*100:.1f}%")

        fold_results.append({
            'fold': fold_i + 1,
            'accuracy': float(acc),
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)
    print(f"  Subject {subject}: {mean_acc*100:.1f}% ± {std_acc*100:.1f}% (5-fold CV)")

    # Final model: train on ALL data, save
    final_model = _create_finetune_model(model, args, device)
    full_ds = TensorDataset(
        torch.from_numpy(np.expand_dims(data, 1)).float(),
        torch.from_numpy(labels).long(),
    )
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False)

    _finetune_loop(
        final_model, data, labels, full_loader, device, args,
        label=f"S{subject:03d}-Full"
    )

    out_dir = args.output_dir / f"subject_{subject:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"model_ft_s{subject:03d}.pth"
    torch.save(final_model, model_path)

    meta = {
        'subject': subject,
        'pretrained': str(args.pretrained),
        'mean_accuracy_5fold': float(mean_acc),
        'std_accuracy_5fold': float(std_acc),
        'folds': fold_results,
        'freeze_cnn': args.freeze_cnn,
        'ft_lr': args.ft_lr,
        'ft_epochs': args.ft_epochs,
        'norm_mean': float(norm_mean),
        'norm_std': float(norm_std),
    }
    with open(model_path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved: {model_path}")

    return {
        'subject': subject,
        'accuracy': float(mean_acc),
        'std': float(std_acc),
        'folds': fold_results,
    }


def _create_finetune_model(base_model, args, device):
    """Clone the pretrained model and optionally freeze CNN layers."""
    import copy
    ft_model = copy.deepcopy(base_model).to(device)

    if args.freeze_cnn:
        for name, param in ft_model.named_parameters():
            if 'cnn' in name:
                param.requires_grad = False

    trainable = sum(p.numel() for p in ft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in ft_model.parameters())
    if args.freeze_cnn:
        print(f"    Trainable: {trainable:,} / {total:,} (CNN frozen)")

    return ft_model


def _finetune_loop(model, X_train, y_train, test_loader, device, args, label=""):
    """Fine-tune training loop with heavy augmentation."""
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.ft_lr, weight_decay=0.02)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.ft_epochs, eta_min=args.ft_lr * 0.01)
    criterion = nn.CrossEntropyLoss().to(device)

    best_acc = 0.0
    best_state = None
    best_ep = 0
    no_improve = 0

    for epoch in range(args.ft_epochs):
        model.train()

        aug_d, aug_l = interaug(X_train, y_train, N_CLASSES, n_aug=4, n_seg=8, batch_size=32)
        ed = np.concatenate([X_train, aug_d])
        el = np.concatenate([y_train, aug_l])
        idx = np.random.permutation(len(ed))
        ed, el = ed[idx], el[idx]

        ds = TensorDataset(
            torch.from_numpy(np.expand_dims(ed, 1)).float(),
            torch.from_numpy(el).long(),
        )
        loader = DataLoader(ds, batch_size=32, shuffle=True)

        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                _, logits = model(bx)
                correct += (logits.argmax(1) == by).sum().item()
                total += bx.size(0)

        acc = correct / max(total, 1)

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 50 == 0:
            print(f"    [{label}] Ep {epoch+1}/{args.ft_epochs} acc={acc:.4f} best={best_acc:.4f}@{best_ep+1}")

        if no_improve >= args.ft_patience:
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, best_acc


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune CTNet on specific subjects")

    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to pretrained model (.pth)")
    p.add_argument("--subjects", type=int, nargs="+", required=True,
                   help="Subjects to fine-tune on")
    p.add_argument("--freeze-cnn", action="store_true", default=True,
                   help="Freeze CNN backbone (default: True)")
    p.add_argument("--no-freeze-cnn", dest="freeze_cnn", action="store_false")
    p.add_argument("--ft-lr", type=float, default=0.0003,
                   help="Fine-tune learning rate (default: 0.0003)")
    p.add_argument("--ft-epochs", type=int, default=200,
                   help="Fine-tune epochs (default: 200)")
    p.add_argument("--ft-patience", type=int, default=50,
                   help="Fine-tune patience (default: 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "physionet_ctnet_finetune")
    p.add_argument("--channel-indices", type=int, nargs="*", default=None,
                   help="Channel indices to select (e.g. 2 3 8 10 12 15 19)")

    return p.parse_args()


def main():
    args = parse_args()

    if not MNE_AVAILABLE:
        print("MNE required: pip install mne")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  CTNet Fine-tuning on PhysioNet")
    print("=" * 60)
    print(f"  Pretrained: {args.pretrained}")
    print(f"  Subjects:   {args.subjects}")
    print(f"  Freeze CNN: {args.freeze_cnn}")
    print(f"  FT LR:      {args.ft_lr}")
    print(f"  FT Epochs:  {args.ft_epochs}")
    print(f"  Device:     {device}")

    base_model = torch.load(args.pretrained, map_location=device, weights_only=False)
    base_model.eval()

    meta_path = args.pretrained.with_suffix('.json')
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    norm_mean = meta.get('norm_mean', 0.0)
    norm_std = meta.get('norm_std', 1.0)
    print(f"  Norm: mean={norm_mean:.4f}, std={norm_std:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    # Optional channel selection
    channel_indices = None
    if hasattr(args, 'channel_indices') and args.channel_indices:
        channel_indices = args.channel_indices
        print(f"  Channel selection: {len(channel_indices)} channels (indices: {channel_indices})")

    for sub in args.subjects:
        result = finetune_subject(base_model, sub, args, device, norm_mean, norm_std,
                                  channel_indices=channel_indices)
        if result:
            all_results.append(result)

    if all_results:
        accs = [r['accuracy'] for r in all_results]
        print(f"\n{'='*60}")
        print(f"  Fine-tuning Summary")
        print(f"{'='*60}")
        for r in all_results:
            print(f"  S{r['subject']:03d}: {r['accuracy']*100:.1f}% ± {r['std']*100:.1f}%")
        print(f"  Overall: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")
        print(f"{'='*60}")

        with open(args.output_dir / "finetune_summary.json", 'w') as f:
            json.dump({
                'results': all_results,
                'mean_accuracy': float(np.mean(accs)),
                'pretrained': str(args.pretrained),
            }, f, indent=2)

        # Bar chart
        fig, ax = plt.subplots(figsize=(max(6, len(all_results)*1.2), 5))
        subs = [r['subject'] for r in all_results]
        bars = ax.bar(range(len(subs)), [a*100 for a in accs], color='#2ecc71', alpha=0.85,
                      yerr=[r['std']*100 for r in all_results], capsize=5)
        ax.axhline(y=np.mean(accs)*100, color='blue', ls='--', lw=2,
                   label=f'Mean: {np.mean(accs)*100:.1f}%')
        ax.axhline(y=56.5, color='red', ls=':', alpha=0.6,
                   label='Pretrained: 56.5%')
        ax.axhline(y=25, color='gray', ls=':', alpha=0.4, label='Chance: 25%')
        for bar, a in zip(bars, accs):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
                    f'{a*100:.1f}%', ha='center', fontweight='bold')
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([f'S{s:03d}' for s in subs])
        ax.set_ylabel('Accuracy (%) — 5-Fold CV')
        ax.set_title('CTNet Fine-tuned Accuracy vs Pretrained Baseline', fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.output_dir / "finetune_comparison.png", dpi=200, bbox_inches='tight')
        plt.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
