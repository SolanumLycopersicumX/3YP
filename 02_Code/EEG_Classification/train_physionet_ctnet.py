#!/usr/bin/env python3
"""
Train CTNet (EEGTransformer) on PhysioNet EEGMMIDB (4-class Motor Imagery)

Uses the REAL CTNet architecture from CTNet_model.py with PhysioNet data:
  - 109 subjects, 64 channels, 160 Hz
  - 4 classes: left(0), right(1), hands/up(2), feet/down(3)
  - Data resampled from 720 -> 1000 time points to match CTNet input
  - InterAug data augmentation (segment recombination within class)

Training modes:
  --mode pool   : Pool subjects, random 80/20 split (default)
  --mode loso   : Leave-One-Subject-Out cross-validation
  --mode subject: Per-subject train/test

Full 109-subject training:
  python scripts/train_physionet_ctnet.py --subjects $(seq 1 109) --epochs 500 --batch-size 128
"""

from __future__ import annotations

import argparse
import sys
import os
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
from sklearn.model_selection import train_test_split
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
    load_multiple_subjects_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)

TARGET_SAMPLES = 1000
DATASET_TYPE = 'P'
N_CLASSES = 4

# Known problematic subjects in PhysioNet EEGMMIDB
SKIP_SUBJECTS = {88, 92, 100, 104}


def resample_epochs(data: np.ndarray, target_len: int = TARGET_SAMPLES) -> np.ndarray:
    """Resample time axis from current length to target_len (720 -> 1000)."""
    n_epochs, n_ch, n_times = data.shape
    if n_times == target_len:
        return data
    resampled = np.zeros((n_epochs, n_ch, target_len), dtype=np.float32)
    for i in range(n_epochs):
        resampled[i] = resample(data[i], target_len, axis=1)
    return resampled


def interaug(data: np.ndarray, labels: np.ndarray, n_classes: int = N_CLASSES,
             n_aug: int = 2, n_seg: int = 8, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    CTNet InterAug: segment recombination within each class.
    Matches CTNet_model.py ExP.interaug() logic.

    For each class, randomly pick segments from different trials of that class
    and stitch them together to form augmented trials.
    """
    n_ch = data.shape[1]
    n_times = data.shape[2]
    seg_len = n_times // n_seg

    aug_data_list = []
    aug_label_list = []

    desired = max(1, n_aug * max(1, batch_size // n_classes))

    for cls in range(n_classes):
        cls_mask = labels == cls
        cls_data = data[cls_mask]

        if len(cls_data) == 0:
            continue

        n_records = min(desired, len(cls_data))
        tmp = np.zeros((n_records, n_ch, n_times), dtype=np.float32)

        for ri in range(n_records):
            for rj in range(n_seg):
                src_idx = np.random.randint(0, len(cls_data))
                s = rj * seg_len
                e = s + seg_len
                tmp[ri, :, s:e] = cls_data[src_idx, :, s:e]

        aug_data_list.append(tmp)
        aug_label_list.append(np.full(n_records, cls, dtype=np.int64))

    aug_data = np.concatenate(aug_data_list)
    aug_labels = np.concatenate(aug_label_list)

    shuffle = np.random.permutation(len(aug_data))
    return aug_data[shuffle], aug_labels[shuffle]


def load_physionet_subjects(
    subjects: List[int],
    bandpass: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, filter, and resample PhysioNet data for multiple subjects."""
    valid_subjects = [s for s in subjects if s not in SKIP_SUBJECTS]
    skipped = [s for s in subjects if s in SKIP_SUBJECTS]

    if skipped:
        print(f"  Skipping known problematic subjects: {skipped}")

    print(f"\n{'='*60}")
    print(f"Loading PhysioNet 4-class MI data")
    print(f"  Subjects: {len(valid_subjects)} / {len(subjects)} requested")
    print(f"  Bandpass: {'8-30 Hz' if bandpass else 'off'}")
    print(f"{'='*60}")

    all_data, all_labels, all_sids = [], [], []

    for sub in valid_subjects:
        try:
            if verbose:
                print(f"  Loading S{sub:03d}...", end=" ", flush=True)
            d, l = load_subject_4class_mne(sub)
            all_data.append(d)
            all_labels.append(l)
            all_sids.append(np.full(len(l), sub))
            if verbose:
                print(f"OK  ({len(l)} epochs)")
        except Exception as e:
            if verbose:
                print(f"FAILED ({e})")

    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_sids)

    loaded_subjects = sorted(set(subject_ids.tolist()))
    print(f"\n  Loaded: {len(loaded_subjects)} subjects, {len(data)} total epochs")

    if bandpass:
        print("  Applying 8-30 Hz bandpass filter...")
        data = preprocess_data(data)

    n_orig = data.shape[2]
    print(f"  Resampling: {n_orig} -> {TARGET_SAMPLES} time points...")
    data = resample_epochs(data, TARGET_SAMPLES)

    print(f"  Final shape: {data.shape}")
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Labels: {dict(zip(unique.tolist(), counts.tolist()))}")

    return data, labels, subject_ids


def train_pool_mode(subjects: List[int], args, device: torch.device):
    """Pool all subjects, random 80/20 split, train one model."""

    data, labels, subject_ids = load_physionet_subjects(
        subjects, bandpass=args.filter, verbose=True
    )

    X_train, X_test, y_train, y_test, sid_train, sid_test = train_test_split(
        data, labels, subject_ids, test_size=0.2, stratify=labels, random_state=42
    )

    mean, std = X_train.mean(), X_train.std()
    X_train_norm = (X_train - mean) / (std + 1e-8)
    X_test_norm = (X_test - mean) / (std + 1e-8)

    print(f"\n  Train: {len(X_train_norm)}, Test: {len(X_test_norm)}")

    model = _build_model(args, 64, device)

    test_ds = TensorDataset(
        torch.from_numpy(np.expand_dims(X_test_norm, 1)).float(),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    best_model, history = _train_loop(
        model, X_train_norm, y_train, test_loader, device, args, label="Pool"
    )

    out_dir = args.output_dir / "pool"
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded_subs = sorted(set(subject_ids.tolist()))
    _save_model(best_model, out_dir / "model_pool.pth", args, loaded_subs, mean, std)
    _evaluate_and_save(best_model, test_loader, device, out_dir, "pool", y_test)
    _plot_history(history, out_dir, "pool")

    with open(out_dir / "norm_params.json", "w") as f:
        json.dump({'mean': float(mean), 'std': float(std)}, f)

    # Per-subject accuracy on test set
    test_subs = sorted(set(sid_test.tolist()))
    _per_subject_eval(best_model, X_test_norm, y_test, sid_test, test_subs,
                      device, out_dir, "pool_per_subject")

    return history


def train_loso_mode(subjects: List[int], args, device: torch.device):
    """Leave-One-Subject-Out cross-validation."""

    data, labels, subject_ids = load_physionet_subjects(
        subjects, bandpass=args.filter, verbose=True
    )

    loaded_subs = sorted(set(subject_ids.tolist()))
    results = []

    for test_sub in loaded_subs:
        print(f"\n{'='*60}")
        print(f"LOSO: Test subject = {test_sub}")
        print(f"{'='*60}")

        test_mask = subject_ids == test_sub
        train_mask = ~test_mask

        X_train, y_train = data[train_mask], labels[train_mask]
        X_test, y_test = data[test_mask], labels[test_mask]

        if len(X_test) == 0:
            continue

        mean, std = X_train.mean(), X_train.std()
        X_train_n = (X_train - mean) / (std + 1e-8)
        X_test_n = (X_test - mean) / (std + 1e-8)

        model = _build_model(args, 64, device)

        test_ds = TensorDataset(
            torch.from_numpy(np.expand_dims(X_test_n, 1)).float(),
            torch.from_numpy(y_test).long(),
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)

        best_model, history = _train_loop(
            model, X_train_n, y_train, test_loader, device, args,
            label=f"LOSO-S{test_sub:03d}"
        )

        out_dir = args.output_dir / "loso" / f"subject_{test_sub:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        _save_model(best_model, out_dir / f"model_s{test_sub:03d}.pth",
                     args, loaded_subs, mean, std)
        metrics = _evaluate_and_save(best_model, test_loader, device,
                                      out_dir, f"s{test_sub:03d}", y_test)
        _plot_history(history, out_dir, f"s{test_sub:03d}")

        results.append({'subject': int(test_sub), **metrics})

    if results:
        _save_summary(results, args.output_dir / "loso")

    return results


def train_subject_mode(subjects: List[int], args, device: torch.device):
    """Per-subject train/test (80/20 split within each subject)."""

    results = []
    for sub in subjects:
        if sub in SKIP_SUBJECTS:
            print(f"  Skipping S{sub:03d} (known problematic)")
            continue

        print(f"\n{'='*60}")
        print(f"Subject-Dependent: Subject {sub}")
        print(f"{'='*60}")

        try:
            sub_data, sub_labels = load_subject_4class_mne(sub)
        except Exception as e:
            print(f"  Failed: {e}")
            continue

        if args.filter:
            sub_data = preprocess_data(sub_data)
        sub_data = resample_epochs(sub_data, TARGET_SAMPLES)

        X_train, X_test, y_train, y_test = train_test_split(
            sub_data, sub_labels, test_size=0.2, stratify=sub_labels, random_state=42
        )

        mean, std = X_train.mean(), X_train.std()
        X_train_n = (X_train - mean) / (std + 1e-8)
        X_test_n = (X_test - mean) / (std + 1e-8)

        model = _build_model(args, 64, device)

        test_ds = TensorDataset(
            torch.from_numpy(np.expand_dims(X_test_n, 1)).float(),
            torch.from_numpy(y_test).long(),
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                 num_workers=0, pin_memory=True)

        best_model, history = _train_loop(
            model, X_train_n, y_train, test_loader, device, args,
            label=f"S{sub:03d}"
        )

        out_dir = args.output_dir / "subject" / f"subject_{sub:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        _save_model(best_model, out_dir / f"model_s{sub:03d}.pth", args, [sub], mean, std)
        metrics = _evaluate_and_save(best_model, test_loader, device,
                                      out_dir, f"s{sub:03d}", y_test)
        _plot_history(history, out_dir, f"s{sub:03d}")
        results.append({'subject': sub, **metrics})

    if results:
        _save_summary(results, args.output_dir / "subject")

    return results


# ============================================================================
# Internal helpers
# ============================================================================

def _build_model(args, n_channels: int, device: torch.device) -> EEGTransformer:
    """Build EEGTransformer with PhysioNet-appropriate parameters."""
    model = EEGTransformer(
        heads=args.heads,
        emb_size=args.emb_size,
        depth=args.depth,
        database_type=DATASET_TYPE,
        eeg1_f1=args.f1,
        eeg1_kernel_size=args.kernel_size,
        eeg1_D=args.D,
        eeg1_pooling_size1=args.pool1,
        eeg1_pooling_size2=args.pool2,
        eeg1_dropout_rate=args.dropout,
        eeg1_number_channel=n_channels,
        flatten_eeg1=args.flatten,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  Model: CTNet (EEGTransformer)")
    print(f"    heads={args.heads}, depth={args.depth}, emb_size={args.emb_size}")
    print(f"    f1={args.f1}, D={args.D}, kernel={args.kernel_size}")
    print(f"    pool=({args.pool1},{args.pool2}), flatten={args.flatten}")
    print(f"    channels={n_channels}, classes={N_CLASSES}")
    print(f"    Parameters: {n_params:,}")

    return model


def _train_loop(
    model: EEGTransformer,
    X_train: np.ndarray,
    y_train: np.ndarray,
    test_loader: DataLoader,
    device: torch.device,
    args,
    label: str = "",
) -> Tuple[EEGTransformer, dict]:
    """
    Training loop with InterAug data augmentation and early stopping.
    X_train: (N, C, T) normalized training data (no channel dim expansion yet).
    """

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=args.lr * 0.01
    )

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    best_epoch = 0
    epochs_no_improve = 0

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
    }

    print(f"\n  Training: {args.epochs} epochs, patience={args.patience}, "
          f"lr={args.lr}, augment={args.augment}")
    print(f"  Optimizer: AdamW (weight_decay=0.01)")
    print(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=50, T_mult=2)")
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- Build training batch with augmentation ---
        model.train()

        if args.augment:
            aug_data, aug_labels = interaug(
                X_train, y_train, N_CLASSES,
                n_aug=args.n_aug, n_seg=args.n_seg,
                batch_size=args.batch_size,
            )
            epoch_data = np.concatenate([X_train, aug_data])
            epoch_labels = np.concatenate([y_train, aug_labels])
        else:
            epoch_data = X_train
            epoch_labels = y_train

        shuffle_idx = np.random.permutation(len(epoch_data))
        epoch_data = epoch_data[shuffle_idx]
        epoch_labels = epoch_labels[shuffle_idx]

        epoch_data_t = torch.from_numpy(np.expand_dims(epoch_data, 1)).float()
        epoch_labels_t = torch.from_numpy(epoch_labels).long()

        train_ds = TensorDataset(epoch_data_t, epoch_labels_t)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=0, pin_memory=True)

        train_loss, train_correct, train_total = 0.0, 0, 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            _, logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_x.size(0)

        avg_train_loss = train_loss / max(train_total, 1)
        avg_train_acc = train_correct / max(train_total, 1)

        # --- Eval ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                _, logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_x.size(0)

        avg_val_loss = val_loss / max(val_total, 1)
        avg_val_acc = val_correct / max(val_total, 1)

        scheduler.step()

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)

        # --- Early stopping based on val loss ---
        improved = avg_val_loss < best_val_loss - 1e-4
        if improved:
            best_val_loss = avg_val_loss
            best_val_acc = avg_val_acc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # --- Checkpoint every 50 epochs ---
        if (epoch + 1) % 50 == 0 and best_state is not None:
            ckpt_path = args.output_dir / f"checkpoint_ep{epoch+1}.pth"
            torch.save(best_state, ckpt_path)

        if (epoch + 1) % 10 == 0 or improved or (epoch + 1) == args.epochs:
            marker = " *" if improved else ""
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  [{label}] Ep {epoch+1:4d}/{args.epochs} | "
                  f"Train loss={avg_train_loss:.4f} acc={avg_train_acc:.4f} | "
                  f"Val loss={avg_val_loss:.4f} acc={avg_val_acc:.4f} | "
                  f"best={best_epoch+1}({best_val_acc:.4f}) lr={cur_lr:.6f}{marker}")

        if epochs_no_improve >= args.patience:
            print(f"  Early stop at epoch {epoch+1} "
                  f"(best: ep {best_epoch+1}, val_acc={best_val_acc:.4f})")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    print(f"  Training time: {datetime.timedelta(seconds=int(elapsed))}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    history['best_epoch'] = best_epoch
    history['best_val_acc'] = best_val_acc

    return model, history


def _save_model(model, path: Path, args, subjects, mean, std):
    """Save full model (torch.save) for compatibility with existing loading code."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)

    meta = {
        'dataset': 'PhysioNet_EEGMMIDB',
        'dataset_type': DATASET_TYPE,
        'n_channels': 64,
        'n_times': TARGET_SAMPLES,
        'n_classes': N_CLASSES,
        'class_map': {'0': 'left', '1': 'right', '2': 'hands/up', '3': 'feet/down'},
        'heads': args.heads, 'depth': args.depth, 'emb_size': args.emb_size,
        'f1': args.f1, 'D': args.D, 'kernel_size': args.kernel_size,
        'pool1': args.pool1, 'pool2': args.pool2,
        'flatten': args.flatten, 'dropout': args.dropout,
        'n_subjects': len(subjects),
        'subjects': subjects,
        'norm_mean': float(mean),
        'norm_std': float(std),
        'augment': args.augment,
        'n_aug': args.n_aug,
        'n_seg': args.n_seg,
    }
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  Model saved: {path}")


def _evaluate_and_save(model, test_loader, device, out_dir, tag, y_true_all) -> dict:
    """Run final evaluation and save confusion matrix + metrics."""
    from sklearn.metrics import confusion_matrix, classification_report

    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            _, logits = model(batch_x)
            all_preds.append(logits.argmax(dim=1).cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = y_true_all

    acc, prec, rec, f1, kappa = calMetrics(y_true, y_pred)

    class_names = ['Left', 'Right', 'Hands/Up', 'Feet/Down']
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    metrics = {
        'accuracy': float(acc), 'precision': float(prec),
        'recall': float(rec), 'f1': float(f1), 'kappa': float(kappa),
    }

    print(f"\n  Results ({tag}):")
    print(f"    Accuracy:  {acc*100:.2f}%  |  Kappa: {kappa:.4f}")
    print(f"    Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1: {f1*100:.2f}%")

    for i, name in enumerate(class_names):
        cd = report.get(name, {})
        print(f"    {name:12s}  P={cd.get('precision',0)*100:5.1f}%  "
              f"R={cd.get('recall',0)*100:5.1f}%  F1={cd.get('f1-score',0)*100:5.1f}%  "
              f"n={int(cd.get('support',0))}")

    with open(out_dir / f"metrics_{tag}.json", 'w') as f:
        json.dump({'metrics': metrics, 'report': report}, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label='%')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            c = 'white' if cm_pct[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_pct[i,j]:.1f}%\n({cm[i,j]})',
                    ha='center', va='center', fontsize=10, color=c, fontweight='bold')
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(class_names, fontsize=11)
    ax.set_yticklabels(class_names, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('True', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix — {tag}\nAccuracy: {acc:.2%}  |  Kappa: {kappa:.4f}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_matrix_{tag}.png", dpi=200, bbox_inches='tight')
    plt.close()

    return metrics


def _per_subject_eval(model, X_test, y_test, sid_test, subjects, device, out_dir, tag):
    """Per-subject accuracy on test set."""
    results = []
    for sub in subjects:
        mask = sid_test == sub
        if mask.sum() == 0:
            continue
        yt = y_test[mask]
        ds = TensorDataset(
            torch.from_numpy(np.expand_dims(X_test[mask], 1)).float(),
            torch.from_numpy(yt).long(),
        )
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in loader:
                _, logits = model(bx.to(device))
                preds.append(logits.argmax(1).cpu().numpy())
        yp = np.concatenate(preds)
        acc = (yt == yp).mean()
        results.append({'subject': int(sub), 'accuracy': float(acc), 'n': int(mask.sum())})

    if results:
        accs = [r['accuracy'] for r in results]
        with open(out_dir / f"{tag}.json", 'w') as f:
            json.dump({
                'results': results,
                'mean_acc': float(np.mean(accs)),
                'std_acc': float(np.std(accs)),
            }, f, indent=2)

        fig, ax = plt.subplots(figsize=(max(10, len(results)*0.3), 5))
        subs = [r['subject'] for r in results]
        ax.bar(range(len(subs)), [a*100 for a in accs], color='steelblue', alpha=0.8, width=0.8)
        ax.axhline(y=np.mean(accs)*100, color='red', linestyle='--', lw=2,
                   label=f'Mean: {np.mean(accs)*100:.1f}%')
        ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Chance 25%')
        ax.set_xticks(range(len(subs)))
        ax.set_xticklabels([f'{s}' for s in subs], rotation=90, fontsize=6)
        ax.set_xlabel('Subject')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Per-Subject Test Accuracy ({tag})\n'
                     f'Mean: {np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%',
                     fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{tag}.png", dpi=200, bbox_inches='tight')
        plt.close()

        print(f"\n  Per-subject accuracy ({len(results)} subjects): "
              f"{np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%")


def _plot_history(history: dict, out_dir: Path, tag: str):
    """Plot training curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-', label='Train', lw=2, alpha=0.8)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val', lw=2, alpha=0.8)
    if 'best_epoch' in history:
        ax1.axvline(x=history['best_epoch']+1, color='green', ls='--', alpha=0.5,
                    label=f"Best (ep {history['best_epoch']+1})")
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss — {tag}', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-', label='Train', lw=2, alpha=0.8)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val', lw=2, alpha=0.8)
    if 'best_epoch' in history:
        ax2.axvline(x=history['best_epoch']+1, color='green', ls='--', alpha=0.5,
                    label=f"Best (ep {history['best_epoch']+1})")
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Accuracy — {tag}', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3); ax2.set_ylim([0, 1])

    fig.suptitle(f'CTNet PhysioNet — {tag}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / f"training_curve_{tag}.png", dpi=150, bbox_inches='tight')
    plt.close()


def _save_summary(results: list, out_dir: Path):
    """Save cross-subject summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    accs = [r['accuracy'] for r in results]
    kappas = [r['kappa'] for r in results]

    summary = {
        'results': results,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_kappa': float(np.mean(kappas)),
        'std_kappa': float(np.std(kappas)),
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Summary ({len(results)} subjects)")
    print(f"  Accuracy: {np.mean(accs)*100:.2f}% +/- {np.std(accs)*100:.2f}%")
    print(f"  Kappa:    {np.mean(kappas):.4f} +/- {np.std(kappas):.4f}")
    print(f"{'='*60}")

    fig, ax = plt.subplots(figsize=(max(8, len(results)*0.4), 5))
    subs = [r['subject'] for r in results]
    ax.bar(range(len(subs)), [a*100 for a in accs], color='steelblue', alpha=0.8)
    ax.axhline(y=np.mean(accs)*100, color='red', ls='--', lw=2,
               label=f'Mean: {np.mean(accs)*100:.1f}%')
    ax.axhline(y=25, color='gray', ls=':', alpha=0.5, label='Chance (25%)')
    ax.set_xticks(range(len(subs)))
    ax.set_xticklabels([f'S{s:03d}' for s in subs], rotation=90, fontsize=6)
    ax.set_xlabel('Subject'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('CTNet PhysioNet — Per-Subject Accuracy', fontweight='bold')
    ax.legend(); ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_summary.png", dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train CTNet on PhysioNet EEGMMIDB (4-class MI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                   help="Subject IDs (default: 1-10)")
    p.add_argument("--mode", choices=['pool', 'loso', 'subject'], default='pool',
                   help="Training mode (default: pool)")
    p.add_argument("--filter", action="store_true", default=True,
                   help="Apply 8-30 Hz bandpass (default: True)")
    p.add_argument("--no-filter", dest="filter", action="store_false")

    g = p.add_argument_group("Model architecture")
    g.add_argument("--heads", type=int, default=2)
    g.add_argument("--depth", type=int, default=6)
    g.add_argument("--emb-size", type=int, default=40)
    g.add_argument("--f1", type=int, default=20)
    g.add_argument("--D", type=int, default=2)
    g.add_argument("--kernel-size", type=int, default=64)
    g.add_argument("--pool1", type=int, default=8)
    g.add_argument("--pool2", type=int, default=8)
    g.add_argument("--flatten", type=int, default=600)
    g.add_argument("--dropout", type=float, default=0.3)

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=800)
    g.add_argument("--batch-size", type=int, default=128)
    g.add_argument("--lr", type=float, default=0.0005)
    g.add_argument("--patience", type=int, default=120)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--augment", action="store_true", default=True,
                   help="Enable InterAug data augmentation (default: True)")
    g.add_argument("--no-augment", dest="augment", action="store_false")
    g.add_argument("--n-aug", type=int, default=3,
                   help="Augmentation multiplier (default: 3)")
    g.add_argument("--n-seg", type=int, default=8,
                   help="Number of segments for InterAug (default: 8)")

    g = p.add_argument_group("Output")
    g.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "physionet_ctnet",
                   help="Output directory")

    return p.parse_args()


def main():
    args = parse_args()

    if not MNE_AVAILABLE:
        print("MNE is required: pip install mne")
        sys.exit(1)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  CTNet Training on PhysioNet EEGMMIDB")
    print("=" * 60)
    print(f"  Mode:      {args.mode}")
    print(f"  Subjects:  {len(args.subjects)} ({min(args.subjects)}-{max(args.subjects)})")
    print(f"  Device:    {device}")
    print(f"  Epochs:    {args.epochs}")
    print(f"  Batch:     {args.batch_size}")
    print(f"  LR:        {args.lr}")
    print(f"  Patience:  {args.patience}")
    print(f"  Filter:    {'8-30 Hz' if args.filter else 'off'}")
    print(f"  Augment:   {args.augment} (n_aug={args.n_aug}, n_seg={args.n_seg})")
    print(f"  Output:    {args.output_dir}")
    print("=" * 60)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    if args.mode == 'pool':
        train_pool_mode(args.subjects, args, device)
    elif args.mode == 'loso':
        train_loso_mode(args.subjects, args, device)
    elif args.mode == 'subject':
        train_subject_mode(args.subjects, args, device)

    total = time.time() - t0
    print(f"\nTotal time: {datetime.timedelta(seconds=int(total))}")
    print("Done!")


if __name__ == "__main__":
    main()
