#!/usr/bin/env python3
"""Visualise CTNet training logs exported by CTNet_model.py.

Reads the per-subject sheets inside `process_train.xlsx` and generates
accuracy/loss curves saved as PNG files. Designed for headless use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import matplotlib

# Force non-interactive backend for servers/headless environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


CurveColumns = Mapping[str, Sequence[float]]


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CTNet training curves.")
    parser.add_argument(
        "--process-file",
        required=True,
        help="Path to process_train.xlsx exported by the training script.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Subject sheet names/indices to plot (default: all sheets).",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to store PNG files (default: alongside process file).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Image resolution for saved figures (default: 150).",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=10.0,
        help="Figure width in inches (default: 10).",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=4.0,
        help="Figure height in inches (default: 4).",
    )
    return parser.parse_args(list(argv))


def load_sheets(process_file: Path) -> Mapping[str, pd.DataFrame]:
    sheets = pd.read_excel(process_file, sheet_name=None)
    cleaned = {}
    for name, df in sheets.items():
        df = df.copy()
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        cleaned[name] = df
    return cleaned


def select_subjects(
    available: Sequence[str], requested: Sequence[str] | None
) -> Sequence[str]:
    if not requested:
        return list(available)
    wanted = []
    for subject in requested:
        if subject in available:
            wanted.append(subject)
            continue
        # Allow numeric indices such as "1" or 1.
        if subject.isdigit() and subject in available:
            wanted.append(subject)
            continue
        raise ValueError(f"Subject '{subject}' not found in workbook sheets {available}")
    return wanted


def extract_curves(df: pd.DataFrame) -> CurveColumns:
    required = {"epoch", "val_acc", "train_acc", "val_loss", "train_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in DataFrame")
    return {
        "epoch": df["epoch"].to_numpy(),
        "train_acc": df["train_acc"].to_numpy(),
        "val_acc": df["val_acc"].to_numpy(),
        "train_loss": df["train_loss"].to_numpy(),
        "val_loss": df["val_loss"].to_numpy(),
    }


def plot_curves(
    subject: str,
    curves: CurveColumns,
    output_dir: Path,
    width: float,
    height: float,
    dpi: int,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(width, height))
    epochs = curves["epoch"]

    axes[0].plot(epochs, curves["train_acc"], label="Train acc", marker="o")
    axes[0].plot(epochs, curves["val_acc"], label="Val acc", marker="s")
    axes[0].set_title(f"Subject {subject} Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", linewidth=0.5)
    axes[0].legend()

    axes[1].plot(epochs, curves["train_loss"], label="Train loss", marker="o")
    axes[1].plot(epochs, curves["val_loss"], label="Val loss", marker="s")
    axes[1].set_title(f"Subject {subject} Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Cross-entropy loss")
    axes[1].grid(True, linestyle="--", linewidth=0.5)
    axes[1].legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"subject_{subject}_curves.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    process_path = Path(args.process_file).expanduser().resolve()
    if not process_path.exists():
        raise FileNotFoundError(f"process file not found: {process_path}")

    sheets = load_sheets(process_path)
    subjects = select_subjects(list(sheets.keys()), args.subjects)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else process_path.parent / "plots"
    )

    for subject in subjects:
        curves = extract_curves(sheets[subject])
        out_path = plot_curves(
            subject=subject,
            curves=curves,
            output_dir=output_dir,
            width=args.width,
            height=args.height,
            dpi=args.dpi,
        )
        print(f"[OK] saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
