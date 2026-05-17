from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from models import CLASS_NAMES


def _selected_channel_indices(n_channels: int, max_channels: int) -> list[int]:
    return list(range(min(n_channels, max_channels)))


def make_eeg_figure(
    raw_eeg,
    preprocessed_eeg,
    sampling_rate,
    channel_names,
    max_channels=8,
    channel_height=1.15,
):
    raw = np.asarray(raw_eeg)
    preprocessed = np.asarray(preprocessed_eeg)
    if raw.ndim != 2 or preprocessed.ndim != 2:
        raise ValueError("raw_eeg and preprocessed_eeg must both be 2D arrays")
    if raw.shape != preprocessed.shape:
        raise ValueError("raw_eeg and preprocessed_eeg shapes must match")
    if raw.shape[0] <= 0 or raw.shape[1] <= 0:
        raise ValueError("raw_eeg and preprocessed_eeg must have positive channels and samples")
    if max_channels <= 0:
        raise ValueError("max_channels must be positive")

    indices = _selected_channel_indices(raw.shape[0], max_channels)
    fig_height = max(2.8, len(indices) * float(channel_height) + 0.9)
    fig, axes = plt.subplots(
        len(indices),
        1,
        sharex=True,
        squeeze=False,
        figsize=(8.5, fig_height),
    )
    time = np.arange(raw.shape[1], dtype=float) / float(sampling_rate)
    raw_uv = raw * 1e6
    preprocessed_uv = preprocessed * 1e6

    for axis, channel_index in zip(axes[:, 0], indices):
        name = (
            channel_names[channel_index]
            if channel_index < len(channel_names)
            else f"Channel {channel_index}"
        )
        raw_line = axis.plot(time, raw_uv[channel_index], label="Raw", linewidth=1.0)
        processed_line = axis.plot(
            time,
            preprocessed_uv[channel_index],
            label="Preprocessed",
            linewidth=1.0,
        )
        axis.set_ylabel(f"{name} (uV)")
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.legend(
        [raw_line[0], processed_line[0]],
        ["Raw", "Preprocessed"],
        loc="upper right",
        ncol=2,
        bbox_to_anchor=(0.98, 0.995),
    )
    fig.suptitle("EEG Traces", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig


def make_trajectory_figure(trajectory_yz):
    fig, axis = plt.subplots()
    points = np.asarray(trajectory_yz, dtype=float)

    if points.size:
        axis.plot(points[:, 0], points[:, 1], marker="o")
        axis.scatter(points[-1, 0], points[-1, 1], marker="x", color="red", label="Current")
        axis.legend(loc="best")

    axis.set_xlabel("Y")
    axis.set_ylabel("Z")
    axis.set_title("Arm Trajectory")
    fig.tight_layout()
    return fig


def make_probability_figure(probabilities):
    values = np.zeros(len(CLASS_NAMES), dtype=float)
    if probabilities is not None:
        incoming = np.asarray(probabilities, dtype=float).ravel()
        values[: min(len(values), len(incoming))] = incoming[: len(values)]

    fig, axis = plt.subplots()
    axis.bar(CLASS_NAMES, values)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Probability")
    axis.set_title("Class Probabilities")
    fig.tight_layout()
    return fig


def export_jsonl(path: Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True))
            handle.write("\n")


def export_csv(path: Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
