#!/usr/bin/env python3
"""Utility helpers for CTNet Phase 3 automation."""

from __future__ import annotations

import warnings
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd
import json


_COLUMN_ALIASES = {
    "accuray": "accuracy",  # keep legacy column spelling compatible
}


def _normalise_key(key: str) -> str:
    key = str(key).strip()
    return _COLUMN_ALIASES.get(key, key)


def _convert_value(value):
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _convert_record(record: Dict) -> Dict[str, object]:
    return {_normalise_key(k): _convert_value(v) for k, v in record.items()}


def parse_result_metrics(result_dir: Path) -> Optional[Dict[str, object]]:
    """Parse `result_metric.xlsx` located inside `result_dir`.

    Parameters
    ----------
    result_dir : Path
        Directory that should contain a `result_metric.xlsx` file.

    Returns
    -------
    dict | None
        Dictionary with keys `mean`, `std`, `per_subject` when parsing succeeds.
        Returns None if the Excel file is missing, empty, or cannot be read.
    """
    result_dir = Path(result_dir)
    metrics_path = result_dir / "result_metric.xlsx"
    if not metrics_path.exists():
        return None
    if metrics_path.stat().st_size == 0:
        warnings.warn(f"[parse_result_metrics] empty file: {metrics_path}")
        return None

    try:
        df = pd.read_excel(metrics_path, engine="openpyxl")
    except Exception as exc:  # pylint: disable=broad-except
        warnings.warn(f"[parse_result_metrics] failed to read {metrics_path}: {exc}")
        return None

    if df.empty:
        warnings.warn(f"[parse_result_metrics] empty data frame: {metrics_path}")
        return None

    # Copy to avoid modifying original df when slicing
    df = df.copy()
    rows: List[Dict[str, object]] = df.to_dict(orient="records")
    if not rows:
        warnings.warn(f"[parse_result_metrics] no rows detected: {metrics_path}")
        return None

    if len(rows) >= 2:
        mean_row = _convert_record(rows[-2])
        std_row = _convert_record(rows[-1])
        per_subject = [_convert_record(row) for row in rows[:-2]]
    else:
        mean_row = _convert_record(rows[-1])
        std_row = {}
        per_subject = []

    return {
        "mean": mean_row,
        "std": std_row,
        "per_subject": per_subject,
        "source_path": str(metrics_path),
    }


def _load_metadata(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # pragma: no cover - defensive
        return {}


def _merge_record(
    result_dir: Path,
    metrics: Optional[Dict[str, object]],
    metadata: Dict[str, object],
) -> Dict[str, object]:
    record: Dict[str, object] = {"result_dir": result_dir.name}

    config = metadata.get("config", {}) if isinstance(metadata, dict) else {}
    if isinstance(config, dict):
        record.update(
            {
                "type": config.get("type"),
                "evaluate_mode": config.get("evaluate_mode"),
                "epochs": config.get("epochs"),
                "n_subject": config.get("n_subject"),
                "n_aug": config.get("n_aug"),
                "n_seg": config.get("n_seg"),
                "heads": config.get("heads"),
                "depth": config.get("depth"),
                "emb_dim": config.get("emb_dim"),
                "validate_ratio": config.get("validate_ratio"),
                "run_tag": config.get("run_tag"),
                "notes": config.get("notes"),
            }
        )

    if isinstance(metadata, dict):
        if "name" in metadata:
            record["name"] = metadata.get("name")
        if "duration_sec" in metadata:
            record["duration_sec"] = metadata.get("duration_sec")
        if "status" in metadata:
            record["status"] = metadata.get("status")
        if "detected_by" in metadata:
            record["detected_by"] = metadata.get("detected_by")

    if "type" not in record or record["type"] in (None, ""):
        parts = result_dir.name.split("_")
        if len(parts) >= 1:
            record["type"] = parts[0]

    if metrics:
        mean = metrics.get("mean", {})
        std = metrics.get("std", {})
        for key, value in mean.items():
            record[f"mean_{key}"] = value
        for key, value in std.items():
            record[f"std_{key}"] = value
        record["subjects_count"] = len(metrics.get("per_subject", []))
    else:
        record["mean_accuracy"] = None

    return record


def collect_result_records(
    root: Path,
    patterns: Sequence[str],
    include_empty: bool = False,
) -> List[Dict[str, object]]:
    root = Path(root)
    matched: List[Path] = []
    for pattern in patterns:
        matched.extend(root.glob(pattern))
    directories = sorted({p for p in matched if p.is_dir()}, key=lambda p: p.name)

    records: List[Dict[str, object]] = []
    for result_dir in directories:
        metrics = parse_result_metrics(result_dir)
        metadata = _load_metadata(result_dir / "run_metadata.json")

        if not metrics and not include_empty:
            continue

        record = _merge_record(result_dir, metrics, metadata)
        records.append(record)

    return records


def cleanup_result_directories(
    root: Path,
    patterns: Sequence[str],
    archive_root: Optional[Path] = None,
) -> List[Dict[str, str]]:
    """Move directories with missing/empty result metrics to an archive folder."""

    root = Path(root)
    patterns = list(patterns) or ["CTNet_*"]
    moved_info: List[Dict[str, str]] = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_archive = Path(archive_root) if archive_root else root / "archive" / "old_results"
    target_base = base_archive / timestamp

    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_dir():
                continue
            metrics_path = path / "result_metric.xlsx"
            if metrics_path.exists() and metrics_path.stat().st_size > 0:
                continue

            if not target_base.exists():
                target_base.mkdir(parents=True, exist_ok=True)

            target_path = target_base / path.name
            suffix = 1
            while target_path.exists():
                target_path = target_base / f"{path.name}_{suffix}"
                suffix += 1

            shutil.move(str(path), str(target_path))
            moved_info.append({
                "original": str(path),
                "archived": str(target_path),
                "metrics_path": str(metrics_path),
            })

    return moved_info


__all__ = [
    "parse_result_metrics",
    "collect_result_records",
    "cleanup_result_directories",
]
