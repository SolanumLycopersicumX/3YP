#!/usr/bin/env python3
"""Lightweight wrapper to train CTNet and report accuracy.

This avoids editing CTNet_model.py directly. It instantiates ExP for a
single subject and prints the test accuracy plus where results are saved.

Example
-------
python scripts/train_ctnet_quick.py \
    --subject 1 --dataset A --mode subject-dependent \
    --epochs 50 --run-dir CTNet_quick_run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("KMP_AFFINITY", "disabled")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import CTNet_model as ctnet
try:
    torch.serialization.add_safe_globals([ctnet.EEGTransformer])
except AttributeError:
    pass

_orig_torch_load = torch.load


def _torch_load_wrapper(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_wrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick runner for CTNet ExP")
    p.add_argument("--subject", type=int, default=1, help="subject index (1..9)")
    p.add_argument("--dataset", choices=["A", "B"], default="A", help="BCI dataset type")
    p.add_argument("--mode", default="subject-dependent", choices=["subject-dependent", "LOSO", "LOSO-No"], help="evaluation mode")
    p.add_argument("--data-dir", default="./mymat_raw/", help="dataset root directory")
    p.add_argument("--epochs", type=int, default=100, help="training epochs")
    p.add_argument("--run-dir", default=None, help="output directory for results (default auto)")
    # model hyper-params (kept consistent with CTNet_model defaults)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--emb", type=int, default=16)
    p.add_argument("--validate-ratio", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=512, help="training batch size for ExP (default 512)")
    p.add_argument("--flatten", type=int, default=240)
    p.add_argument("--f1", type=int, default=8)
    p.add_argument("--kernel", type=int, default=64)
    p.add_argument("--D", type=int, default=2)
    p.add_argument("--pool1", type=int, default=8)
    p.add_argument("--pool2", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--n-aug", type=int, default=3)
    p.add_argument("--n-seg", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.run_dir or f"CTNet_quick_{args.dataset}_{int(time.time())}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    exp = ctnet.ExP(
        nsub=args.subject,
        data_dir=args.data_dir,
        result_name=out_dir,
        epochs=args.epochs,
        number_aug=args.n_aug,
        number_seg=args.n_seg,
        gpus=None,
        evaluate_mode=args.mode,
        heads=args.heads,
        emb_size=args.emb,
        depth=args.depth,
        dataset_type=args.dataset,
        eeg1_f1=args.f1,
        eeg1_kernel_size=args.kernel,
        eeg1_D=args.D,
        eeg1_pooling_size1=args.pool1,
        eeg1_pooling_size2=args.pool2,
        eeg1_dropout_rate=args.dropout,
        flatten_eeg1=args.flatten,
        batch_size=args.batch_size,
        validate_ratio=args.validate_ratio,
    )
    test_acc, y_true, y_pred, df_proc, best_epoch = exp.train()
    print(f"[CTNet] subject={args.subject} mode={args.mode} dataset={args.dataset}\n"
          f"accuracy={test_acc:.4f} best_epoch={best_epoch} results_dir={out_dir}")


if __name__ == "__main__":
    # ensure device stays as defined in CTNet_model (cpu by default)
    torch.set_num_threads(1)
    main()
