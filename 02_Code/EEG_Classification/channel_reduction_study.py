#!/usr/bin/env python3
"""
EEG Channel Reduction Study for CTNet on PhysioNet

Phase 1: Channel importance analysis (ablation + gradient)
Phase 2: Train & evaluate CTNet with reduced channel sets
         64 → 32 → 16 → 8 → 4 → 2

The 8-channel target is most important (matches OpenBCI Cyton hardware).

Usage:
  # Phase 1: Analyze channel importance
  python scripts/channel_reduction_study.py --phase importance \
    --pretrained outputs/physionet_ctnet_109sub_v2/pool/model_pool.pth

  # Phase 2: Train all reduced-channel models
  python scripts/channel_reduction_study.py --phase train --subjects 1-50

  # Phase 3: Fine-tune 8ch model on specific subjects
  python scripts/channel_reduction_study.py --phase finetune8 \
    --pretrained8 outputs/channel_reduction/8ch/model_pool.pth \
    --subjects 7 48 3
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from scripts.physionet_loader import (
    load_subject_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)

TARGET_SAMPLES = 1000
N_CLASSES = 4
CLASS_NAMES = ['Left', 'Right', 'Hands/Up', 'Feet/Down']

# PhysioNet EEGMMIDB 64 channels (10-10 system, after MNE standardize)
PHYSIONET_64_CHANNELS = [
    'Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.',
    'C5..', 'C3..', 'C1..', 'Cz..', 'C2..', 'C4..', 'C6..',
    'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.',
    'Fp1.', 'Fpz.', 'Fp2.',
    'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.',
    'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..', 'F6..', 'F8..',
    'Ft7.', 'Ft8.',
    'T7..', 'T8..', 'T9..', 'T10.',
    'Tp7.', 'Tp8.',
    'P7..', 'P5..', 'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..',
    'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
    'O1..', 'Oz..', 'O2..', 'Iz..',
]


def get_channel_names_from_data(subject: int = 1) -> List[str]:
    """Extract actual channel names from PhysioNet data via MNE."""
    import mne
    from mne.datasets import eegbci
    from mne.io import read_raw_edf

    raw_fnames = eegbci.load_data(subject, [4])
    raw = read_raw_edf(raw_fnames[0], preload=False, verbose=False)
    eegbci.standardize(raw)
    ch_names = [ch for ch in raw.ch_names
                if ch not in ['STI 014'] and raw.get_channel_types([ch])[0] == 'eeg']
    return ch_names


# Motor cortex channel selections (domain knowledge)
# Names match MNE standardized PhysioNet channel names
CHANNEL_SUBSETS = {
    32: {
        'name': 'motor_cortex_32',
        'channels': [
            'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6',
            'F3', 'F1', 'Fz', 'F2', 'F4',
            'P3', 'P1', 'Pz', 'P2', 'P4',
            'FT7',
        ],
    },
    16: {
        'name': 'motor_cortex_16',
        'channels': [
            'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C3', 'C1', 'Cz', 'C2', 'C4',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'Fz',
        ],
    },
    8: {
        'name': 'motor_cortex_8_openbci',
        'channels': [
            'C3', 'C4', 'Cz',
            'FC3', 'FC4',
            'CP3', 'CP4',
            'FCz',
        ],
    },
    4: {
        'name': 'motor_cortex_4',
        'channels': ['C3', 'C4', 'Cz', 'FCz'],
    },
    2: {
        'name': 'motor_cortex_2',
        'channels': ['C3', 'C4'],
    },
}


def resample_epochs(data, target_len=TARGET_SAMPLES):
    n, c, t = data.shape
    if t == target_len:
        return data
    out = np.zeros((n, c, target_len), dtype=np.float32)
    for i in range(n):
        out[i] = resample(data[i], target_len, axis=1)
    return out


def interaug(data, labels, n_classes=4, n_aug=3, n_seg=8, batch_size=32):
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


def load_subjects_data(subjects: List[int], channel_indices: Optional[List[int]] = None,
                       verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, preprocess, resample, and optionally select channels."""
    all_data, all_labels, all_sids = [], [], []
    for sub in subjects:
        try:
            d, l = load_subject_4class_mne(sub)
            d = preprocess_data(d)
            d = resample_epochs(d, TARGET_SAMPLES)
            if channel_indices is not None:
                d = d[:, channel_indices, :]
            all_data.append(d)
            all_labels.append(l)
            all_sids.append(np.full(len(l), sub))
            if verbose:
                print(f"  S{sub:03d}: {len(l)} epochs, shape {d.shape}")
        except Exception as e:
            if verbose:
                print(f"  S{sub:03d}: FAILED ({e})")
    data = np.concatenate(all_data)
    labels = np.concatenate(all_labels)
    sids = np.concatenate(all_sids)
    return data, labels, sids


def resolve_channel_indices(ch_names_64: List[str], target_channels: List[str]) -> List[int]:
    """Map channel names to indices in the 64-channel array."""
    indices = []
    for tc in target_channels:
        found = False
        for i, ch in enumerate(ch_names_64):
            if ch.lower() == tc.lower():
                indices.append(i)
                found = True
                break
        if not found:
            for i, ch in enumerate(ch_names_64):
                if tc.lower() in ch.lower() or ch.lower() in tc.lower():
                    indices.append(i)
                    found = True
                    break
        if not found:
            print(f"  WARNING: channel '{tc}' not found in {ch_names_64}")
    return sorted(set(indices))


# ============================================================================
# Phase 1: Channel Importance Analysis
# ============================================================================

def run_importance_analysis(args):
    """Ablation-based channel importance using pretrained 64ch model."""
    print("\n" + "=" * 70)
    print("  Phase 1: Channel Importance Analysis (Ablation)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get channel names
    print("\n[1/4] Extracting channel names...")
    ch_names = get_channel_names_from_data(1)
    n_ch = len(ch_names)
    print(f"  Found {n_ch} channels: {ch_names}")

    # Load pretrained model
    print("\n[2/4] Loading pretrained model...")
    import CTNet_model
    import __main__
    for attr_name in dir(CTNet_model):
        if not attr_name.startswith('_'):
            setattr(__main__, attr_name, getattr(CTNet_model, attr_name))

    model = torch.load(args.pretrained, map_location=device, weights_only=False)
    model.eval()

    meta_path = args.pretrained.with_suffix('.json')
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    norm_mean = meta.get('norm_mean', 0.0)
    norm_std = meta.get('norm_std', 1.0)

    # Load test data (10 representative subjects)
    test_subjects = [1, 3, 7, 9, 15, 22, 35, 48, 62, 70]
    print(f"\n[3/4] Loading test data (subjects: {test_subjects})...")
    data, labels, sids = load_subjects_data(test_subjects, verbose=True)
    data = (data - norm_mean) / (norm_std + 1e-8)
    print(f"  Total: {data.shape}")

    # Baseline accuracy
    print("\n[4/4] Running ablation analysis...")
    baseline_acc = _eval_accuracy(model, data, labels, device)
    print(f"  Baseline (64ch): {baseline_acc*100:.2f}%")

    # Ablation: zero out each channel
    importance = {}
    for ch_idx in range(n_ch):
        data_ablated = data.copy()
        data_ablated[:, ch_idx, :] = 0.0
        acc = _eval_accuracy(model, data_ablated, labels, device)
        drop = baseline_acc - acc
        importance[ch_idx] = {
            'name': ch_names[ch_idx],
            'index': ch_idx,
            'acc_without': float(acc),
            'accuracy_drop': float(drop),
        }
        print(f"  Ch {ch_idx:2d} ({ch_names[ch_idx]:>5s}): "
              f"acc={acc*100:.2f}% drop={drop*100:+.2f}%")

    # Sort by importance
    sorted_channels = sorted(importance.values(), key=lambda x: x['accuracy_drop'], reverse=True)

    # Save results
    out_dir = args.output_dir / "importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'baseline_accuracy': float(baseline_acc),
        'channels': sorted_channels,
        'ch_names': ch_names,
        'test_subjects': test_subjects,
    }
    with open(out_dir / "channel_importance.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Generate top-N channel sets from ablation
    print(f"\n--- Top channels by importance ---")
    for i, ch in enumerate(sorted_channels[:20]):
        print(f"  {i+1:2d}. {ch['name']:>5s} (idx={ch['index']:2d}) "
              f"drop={ch['accuracy_drop']*100:+.3f}%")

    top_8_ablation = [ch['name'] for ch in sorted_channels[:8]]
    top_16_ablation = [ch['name'] for ch in sorted_channels[:16]]
    top_32_ablation = [ch['name'] for ch in sorted_channels[:32]]

    ablation_subsets = {
        8: top_8_ablation,
        16: top_16_ablation,
        32: top_32_ablation,
    }

    with open(out_dir / "ablation_top_channels.json", 'w') as f:
        json.dump({
            'top_8': top_8_ablation,
            'top_16': top_16_ablation,
            'top_32': top_32_ablation,
        }, f, indent=2)

    # Plot
    _plot_importance(sorted_channels, ch_names, out_dir)
    _plot_topography(sorted_channels, ch_names, out_dir)

    print(f"\n  Results saved: {out_dir}")
    return results


def _eval_accuracy(model, data, labels, device, batch_size=64):
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(np.expand_dims(data, 1)).float(),
        torch.from_numpy(labels).long(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    with torch.no_grad():
        for bx, by in loader:
            _, logits = model(bx.to(device))
            correct += (logits.argmax(1).cpu() == by).sum().item()
            total += bx.size(0)
    return correct / max(total, 1)


def _plot_importance(sorted_channels, ch_names, out_dir):
    fig, ax = plt.subplots(figsize=(18, 6))
    names = [ch['name'].rstrip('.') for ch in sorted_channels]
    drops = [ch['accuracy_drop'] * 100 for ch in sorted_channels]

    colors = []
    motor_names = {'c3', 'c4', 'c1', 'c2', 'cz', 'c5', 'c6',
                   'fc3', 'fc4', 'fc1', 'fc2', 'fcz', 'fc5', 'fc6',
                   'cp3', 'cp4', 'cp1', 'cp2', 'cpz', 'cp5', 'cp6'}
    for ch in sorted_channels:
        n = ch['name'].rstrip('.').lower()
        if n in motor_names:
            colors.append('#e74c3c')
        else:
            colors.append('#3498db')

    ax.bar(range(len(names)), drops, color=colors, alpha=0.85)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Accuracy Drop (%)')
    ax.set_title('Channel Importance (Ablation Study)\nRed = Motor Cortex | Blue = Other',
                 fontweight='bold')
    ax.axhline(y=0, color='black', ls='-', lw=0.5)
    ax.grid(True, axis='y', alpha=0.3)

    import matplotlib.patches as mpatches
    motor_patch = mpatches.Patch(color='#e74c3c', label='Motor Cortex')
    other_patch = mpatches.Patch(color='#3498db', label='Other Regions')
    ax.legend(handles=[motor_patch, other_patch])

    plt.tight_layout()
    plt.savefig(out_dir / "channel_importance_bar.png", dpi=200, bbox_inches='tight')
    plt.close()


def _plot_topography(sorted_channels, ch_names, out_dir):
    """Plot a simple 2D map of channel importance."""
    try:
        import mne
        info = mne.create_info(
            ch_names=[c.rstrip('.') for c in ch_names],
            sfreq=160,
            ch_types='eeg'
        )
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage, on_missing='ignore')

        importance_arr = np.zeros(len(ch_names))
        ch_map = {ch['name']: ch['accuracy_drop'] for ch in sorted_channels}
        for i, name in enumerate(ch_names):
            importance_arr[i] = ch_map.get(name, 0.0)

        fig, ax = plt.subplots(figsize=(8, 8))
        mne.viz.plot_topomap(importance_arr, info, axes=ax, show=False,
                             cmap='RdYlBu_r', contours=6)
        ax.set_title('Channel Importance Topography\n(Accuracy Drop when Removed)',
                      fontweight='bold', fontsize=12)
        plt.savefig(out_dir / "channel_importance_topo.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"  Topography plot saved")
    except Exception as e:
        print(f"  Topography plot failed: {e}")


# ============================================================================
# Phase 2: Train with reduced channels
# ============================================================================

def run_channel_training(args):
    """Train CTNet with different channel counts."""
    print("\n" + "=" * 70)
    print("  Phase 2: Channel Reduction Training")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get channel names
    ch_names_64 = get_channel_names_from_data(1)

    # Load ablation-based top channels if available
    ablation_path = args.output_dir / "importance" / "ablation_top_channels.json"
    ablation_subsets = {}
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_subsets = json.load(f)
        print(f"  Loaded ablation-based channel rankings")

    # Parse subjects
    subjects = _parse_subjects(args.subjects_str)
    print(f"  Subjects: {len(subjects)} ({subjects[0]}-{subjects[-1]})")

    # Channel configurations to test
    configs = []

    for n_ch in [64, 32, 16, 8, 4, 2]:
        if n_ch == 64:
            configs.append({
                'n_channels': 64,
                'name': 'full_64ch',
                'indices': list(range(64)),
                'ch_names': ch_names_64,
            })
        else:
            domain_info = CHANNEL_SUBSETS[n_ch]
            domain_idx = resolve_channel_indices(ch_names_64, domain_info['channels'])
            configs.append({
                'n_channels': n_ch,
                'name': f"domain_{domain_info['name']}",
                'indices': domain_idx,
                'ch_names': [ch_names_64[i] for i in domain_idx],
            })

            # Also add ablation-based set if available
            abl_key = f'top_{n_ch}'
            if abl_key in ablation_subsets:
                abl_idx = resolve_channel_indices(ch_names_64, ablation_subsets[abl_key])
                if len(abl_idx) == n_ch:
                    configs.append({
                        'n_channels': n_ch,
                        'name': f"ablation_top{n_ch}",
                        'indices': abl_idx,
                        'ch_names': [ch_names_64[i] for i in abl_idx],
                    })

    print(f"\n  Configurations to train: {len(configs)}")
    for cfg in configs:
        print(f"    {cfg['name']:>30s}: {cfg['n_channels']} channels")

    # Load all data once (64ch)
    print(f"\n  Loading data for {len(subjects)} subjects...")
    data_64, labels, sids = load_subjects_data(subjects, verbose=False)
    print(f"  Data: {data_64.shape}, Labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    all_results = []

    for cfg in configs:
        if args.only_nch and cfg['n_channels'] not in args.only_nch:
            continue

        result = _train_config(cfg, data_64, labels, sids, device, args)
        all_results.append(result)

    # Summary
    _print_summary(all_results, args.output_dir)


def _train_config(cfg, data_64, labels, sids, device, args):
    """Train one channel configuration."""
    n_ch = cfg['n_channels']
    name = cfg['name']
    indices = cfg['indices']

    print(f"\n{'='*60}")
    print(f"  Training: {name} ({n_ch} channels)")
    print(f"  Channels: {cfg['ch_names']}")
    print(f"{'='*60}")

    # Select channels
    if n_ch < 64:
        data = data_64[:, indices, :]
    else:
        data = data_64.copy()

    # Normalize
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + 1e-8)

    # Train/test split (80/20 stratified by subject)
    unique_subs = np.unique(sids)
    np.random.seed(42)
    np.random.shuffle(unique_subs)
    n_train_subs = int(0.8 * len(unique_subs))
    train_subs = set(unique_subs[:n_train_subs])
    test_subs = set(unique_subs[n_train_subs:])

    train_mask = np.isin(sids, list(train_subs))
    test_mask = np.isin(sids, list(test_subs))

    X_train, y_train = data[train_mask], labels[train_mask]
    X_test, y_test = data[test_mask], labels[test_mask]

    print(f"  Train: {X_train.shape} ({len(train_subs)} subjects)")
    print(f"  Test:  {X_test.shape} ({len(test_subs)} subjects)")

    # Build model: patch numberClassChannel so EEGTransformer gets correct channel count
    import CTNet_model as _ctm
    _orig_ncc = _ctm.numberClassChannel
    _ctm.numberClassChannel = lambda db_type: (N_CLASSES, n_ch)

    model = EEGTransformer(
        heads=2, emb_size=40, depth=6,
        database_type='P',
        eeg1_f1=20, eeg1_kernel_size=64, eeg1_D=2,
        eeg1_pooling_size1=8, eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.3,
        eeg1_number_channel=n_ch,
        flatten_eeg1=600,
    ).to(device)

    _ctm.numberClassChannel = _orig_ncc

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    criterion = nn.CrossEntropyLoss().to(device)

    test_ds = TensorDataset(
        torch.from_numpy(np.expand_dims(X_test, 1)).float(),
        torch.from_numpy(y_test).long(),
    )
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    best_acc = 0.0
    best_state = None
    best_ep = 0
    no_improve = 0
    history = {'train_loss': [], 'test_acc': []}

    n_epochs = args.epochs
    patience = args.patience

    for epoch in range(n_epochs):
        model.train()

        aug_d, aug_l = interaug(X_train, y_train, N_CLASSES, n_aug=3, n_seg=8, batch_size=32)
        ed = np.concatenate([X_train, aug_d])
        el = np.concatenate([y_train, aug_l])
        idx = np.random.permutation(len(ed))
        ed, el = ed[idx], el[idx]

        ds = TensorDataset(
            torch.from_numpy(np.expand_dims(ed, 1)).float(),
            torch.from_numpy(el).long(),
        )
        loader = DataLoader(ds, batch_size=64, shuffle=True)

        epoch_loss = 0.0
        n_batches = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            _, logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

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

        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['test_acc'].append(acc)

        improved = False
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ep = epoch
            no_improve = 0
            improved = True
        else:
            no_improve += 1

        if (epoch + 1) % 20 == 0 or improved:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"  Ep {epoch+1:4d}/{n_epochs} loss={epoch_loss/max(n_batches,1):.4f} "
                  f"acc={acc:.4f} best={best_acc:.4f}@{best_ep+1} lr={lr_now:.6f}"
                  f"{' *' if improved else ''}")

        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Restore best
    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    # Final eval with per-class breakdown
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            _, logits = model(bx.to(device))
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_true.append(by.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_true)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    final_acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print(f"\n  Final: Acc={final_acc*100:.2f}% P={precision*100:.1f}% "
          f"R={recall*100:.1f}% F1={f1*100:.1f}%")

    for cls in range(N_CLASSES):
        mask_t = y_true == cls
        mask_p = y_pred == cls
        cls_acc = (y_true[mask_t] == y_pred[mask_t]).mean() if mask_t.sum() > 0 else 0
        print(f"    {CLASS_NAMES[cls]:>10s}: {cls_acc*100:.1f}% (n={mask_t.sum()})")

    # Save
    out_dir = args.output_dir / f"{n_ch}ch" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"model_{name}.pth"
    torch.save(model, model_path)

    result = {
        'name': name,
        'n_channels': n_ch,
        'channel_names': cfg['ch_names'],
        'channel_indices': indices,
        'accuracy': float(final_acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'best_epoch': best_ep + 1,
        'total_params': total_params,
        'norm_mean': float(mean),
        'norm_std': float(std),
    }
    with open(model_path.with_suffix('.json'), 'w') as f:
        json.dump(result, f, indent=2)

    # Training curve
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history['train_loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{name} - Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(history['test_acc'])
    ax2.axhline(y=best_acc, color='r', ls='--', alpha=0.5, label=f'Best: {best_acc*100:.1f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{name} - Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f"training_curve_{name}.png", dpi=150)
    plt.close()

    print(f"  Saved: {out_dir}")
    return result


def _print_summary(all_results, output_dir):
    """Print and save comparison summary."""
    if not all_results:
        return

    print(f"\n{'='*70}")
    print(f"  Channel Reduction Summary")
    print(f"{'='*70}")
    print(f"  {'Config':>30s} | {'Ch':>3s} | {'Acc':>7s} | {'F1':>7s} | {'Params':>8s}")
    print(f"  {'-'*30}-+-{'-'*3}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

    for r in sorted(all_results, key=lambda x: (-x['n_channels'], -x['accuracy'])):
        print(f"  {r['name']:>30s} | {r['n_channels']:>3d} | "
              f"{r['accuracy']*100:>6.2f}% | {r['f1']*100:>6.2f}% | "
              f"{r['total_params']:>8,}")

    with open(output_dir / "channel_reduction_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    names = [r['name'] for r in all_results]
    accs = [r['accuracy'] * 100 for r in all_results]
    n_chs = [r['n_channels'] for r in all_results]

    color_map = {64: '#2ecc71', 32: '#3498db', 16: '#9b59b6',
                 8: '#e74c3c', 4: '#f39c12', 2: '#95a5a6'}
    colors = [color_map.get(n, '#7f8c8d') for n in n_chs]

    bars = ax.bar(range(len(names)), accs, color=colors, alpha=0.85)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'{a:.1f}%', ha='center', fontweight='bold', fontsize=9)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([f"{n}\n({c}ch)" for n, c in zip(names, n_chs)],
                       rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('CTNet Accuracy vs Channel Count', fontweight='bold', fontsize=13)
    ax.axhline(y=25, color='gray', ls=':', alpha=0.4, label='Chance (25%)')
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(accs) + 10)

    plt.tight_layout()
    plt.savefig(output_dir / "channel_reduction_comparison.png", dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Comparison plot saved: {output_dir / 'channel_reduction_comparison.png'}")


# ============================================================================
# Utils
# ============================================================================

def _parse_subjects(s: str) -> List[int]:
    """Parse '1-50' or '1,2,3' into list of ints."""
    result = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            result.extend(range(int(a), int(b) + 1))
        else:
            result.append(int(part))
    return sorted(set(result))


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="EEG Channel Reduction Study")

    p.add_argument("--phase", choices=["importance", "train", "all"], required=True)
    p.add_argument("--pretrained", type=Path,
                   default=_ROOT / "outputs" / "physionet_ctnet_109sub_v2" / "pool" / "model_pool.pth")
    p.add_argument("--subjects-str", type=str, default="1-50",
                   help="Subjects for training (e.g. '1-50' or '1,3,7')")
    p.add_argument("--only-nch", type=int, nargs="*", default=None,
                   help="Only train specific channel counts (e.g. --only-nch 8 16)")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "channel_reduction")

    return p.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.phase in ("importance", "all"):
        run_importance_analysis(args)

    if args.phase in ("train", "all"):
        run_channel_training(args)

    print("\nDone!")


if __name__ == "__main__":
    main()
