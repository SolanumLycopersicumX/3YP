#!/usr/bin/env python3
"""Automation helper to launch CTNet Phase 3 experiments sequentially."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from phase3_utils import (
    parse_result_metrics,
    collect_result_records,
    cleanup_result_directories,
)


ENV_MAP = {
    "type": "CTNET_TYPE",
    "evaluate_mode": "CTNET_EVALUATE_MODE",
    "epochs": "CTNET_EPOCHS",
    "n_subject": "CTNET_N_SUBJECT",
    "n_aug": "CTNET_N_AUG",
    "n_seg": "CTNET_N_SEG",
    "heads": "CTNET_HEADS",
    "depth": "CTNET_DEPTH",
    "emb_dim": "CTNET_EMB_DIM",
    "validate_ratio": "CTNET_VALIDATE_RATIO",
    "run_tag": "CTNET_RUN_TAG",
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict):
        if "experiments" in data:
            data = data["experiments"]
        else:
            data = [data]
    if not isinstance(data, list):
        raise ValueError(f"配置文件格式错误: {path}")
    experiments: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"实验条目必须是对象: {item}")
        experiments.append(item)
    return experiments


def _create_experiments_from_args(args) -> List[Dict[str, Any]]:
    exp: Dict[str, Any] = {}
    for key in ENV_MAP.keys():
        value = getattr(args, key, None)
        if value is not None:
            exp[key] = value
    if args.name:
        exp["name"] = args.name
    if args.notes:
        exp["notes"] = args.notes
    return [exp] if exp else []


def _append_csv(row: Dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "status",
        "result_dir",
        "type",
        "evaluate_mode",
        "epochs",
        "n_subject",
        "n_aug",
        "n_seg",
        "heads",
        "depth",
        "emb_dim",
        "validate_ratio",
        "run_tag",
        "duration_sec",
        "mean_accuracy",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "mean_kappa",
        "std_accuracy",
        "std_kappa",
        "command",
        "notes",
        "detected_by",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _prepare_env(base_env: Dict[str, str], cfg: Dict[str, Any], extra_env: Dict[str, str]) -> Dict[str, str]:
    env = base_env.copy()
    for key, var in ENV_MAP.items():
        if key in cfg and cfg[key] is not None:
            env[var] = str(cfg[key])
    env.update(extra_env)
    return env


def _scan_result_dirs(root: Path) -> Dict[str, float]:
    info: Dict[str, float] = {}
    for path in root.iterdir():
        if not path.is_dir():
            continue
        metrics_file = path / "result_metric.xlsx"
        if not metrics_file.exists():
            continue
        try:
            info[path.name] = metrics_file.stat().st_mtime
        except FileNotFoundError:
            continue
    return info


def _create_metadata(
    result_dir: Path,
    cfg: Dict[str, Any],
    env_vars: Dict[str, str],
    metrics: Optional[Dict[str, Any]],
    status: str,
    start_time: float,
    duration: float,
    command: List[str],
    detection: str,
) -> None:
    metadata = {
        "config": dict(cfg),
        "name": cfg.get("name"),
        "notes": cfg.get("notes"),
        "env": env_vars,
        "status": status,
        "start_time": datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
        "end_time": datetime.fromtimestamp(start_time + duration).isoformat(timespec="seconds"),
        "duration_sec": duration,
        "command": command,
        "detected_by": detection,
        "metrics": metrics,
    }
    with (result_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _generate_summary_report(
    root: Path,
    patterns: List[str],
    output: Optional[Path],
    group_by: Optional[List[str]],
    sort_by: Optional[str],
    descending: bool,
    include_empty: bool,
) -> None:
    print("\n=== 自动汇总 ===")
    records = collect_result_records(root, patterns, include_empty=include_empty)
    if not records:
        print("未找到匹配的结果目录。")
        return

    df = pd.DataFrame(records)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)

    print(df.to_string(index=False))

    if group_by:
        missing_cols = [col for col in group_by if col not in df.columns]
        if missing_cols:
            print(f"[警告] 以下分组列不存在：{missing_cols}")
        else:
            agg_cols = [col for col in df.columns if col.startswith("mean_")]
            if agg_cols:
                grouped = df.groupby(group_by)[agg_cols].mean().reset_index()
                print("\n=== 自动分组均值 ===")
                print(grouped.to_string(index=False))

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"\n汇总结果已写入 {output}")


def run_experiment(
    cfg: Dict[str, Any],
    python_exec: Path,
    project_root: Path,
    log_csv: Optional[Path],
    extra_env: Dict[str, str],
    dry_run: bool,
    stop_on_failure: bool,
) -> None:
    base_env = os.environ.copy()
    env = _prepare_env(base_env, cfg, extra_env)
    command = [str(python_exec), "main_subject_specific.py"]

    print("=" * 80)
    print(f"开始实验: {cfg.get('name') or cfg.get('run_tag') or cfg}")
    print(f"命令: {' '.join(command)}")
    ctnet_entries = [f"{k}={env[k]}" for k in sorted(env.keys()) if k.startswith("CTNET_")]
    other_entries = [f"{k}={env[k]}" for k in extra_env]
    print(f"环境变量(CTNET_*): {{ {', '.join(ctnet_entries)} }}")
    if other_entries:
        print(f"环境变量(额外): {{ {', '.join(other_entries)} }}")

    if dry_run:
        print("dry-run 模式，跳过实际执行。")
        return

    before_info = _scan_result_dirs(project_root)
    start_ts = time.time()
    status = "ok"
    try:
        subprocess.run(command, cwd=project_root, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        status = f"failed (exit={exc.returncode})"
        print(f"[警告] 实验执行失败：{exc}")
        if stop_on_failure:
            raise
    duration = time.time() - start_ts
    after_info = _scan_result_dirs(project_root)
    new_names = sorted(set(after_info) - set(before_info))
    updated_names = sorted(
        {
            name
            for name in after_info
            if name in before_info and after_info[name] > before_info[name] + 1e-3
        }
    )

    detected_pairs: List[tuple[str, str]] = []
    detected_pairs.extend((name, "new") for name in new_names)
    detected_pairs.extend(
        (name, "updated") for name in updated_names if name not in new_names
    )

    if not detected_pairs:
        print("[警告] 未检测到新的结果目录，将使用最近修改的目录。")
        if after_info:
            latest_name = max(after_info.items(), key=lambda item: item[1])[0]
            detected_pairs = [(latest_name, "fallback")]
        else:
            detected_pairs = []

    for result_name, detection_mode in detected_pairs or [("<未知>", "none")]:
        result_path = project_root / result_name if result_name != "<未知>" else project_root
        metrics = None
        if result_name != "<未知>":
            metrics = parse_result_metrics(result_path)
            env_subset = {k: env[k] for k in env if k.startswith("CTNET_")}
            for key in extra_env:
                env_subset[key] = env[key]
            try:
                _create_metadata(
                    result_path,
                    cfg,
                    env_subset,
                    metrics,
                    status,
                    start_ts,
                    duration,
                    command,
                    detection_mode,
                )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"[警告] 无法写入 metadata: {exc}")

        mean_accuracy = metrics and metrics["mean"].get("accuracy")
        mean_kappa = metrics and metrics["mean"].get("kappa")
        print(f"结果目录: {result_name}")
        print(f"  检测方式: {detection_mode}")
        if metrics:
            print(f"  平均准确率: {mean_accuracy:.4f}" if mean_accuracy is not None else "  平均准确率: 未知")
            print(f"  平均Kappa: {mean_kappa:.4f}" if mean_kappa is not None else "  平均Kappa: 未知")
        else:
            print("  未能解析 result_metric.xlsx")

        if log_csv:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": status,
                "result_dir": result_name,
                "command": " ".join(command),
                "notes": cfg.get("notes"),
                "duration_sec": round(duration, 2),
                "detected_by": detection_mode,
            }
            for key in ("type", "evaluate_mode", "epochs", "n_subject", "n_aug", "n_seg", "heads", "depth", "emb_dim", "validate_ratio", "run_tag"):
                row[key] = cfg.get(key)
            if metrics:
                mean = metrics["mean"]
                std = metrics["std"]
                row["mean_accuracy"] = mean.get("accuracy")
                row["mean_precision"] = mean.get("precision")
                row["mean_recall"] = mean.get("recall")
                row["mean_f1"] = mean.get("f1")
                row["mean_kappa"] = mean.get("kappa")
                row["std_accuracy"] = std.get("accuracy")
                row["std_kappa"] = std.get("kappa")
            _append_csv(row, log_csv)


def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行 main_subject_specific.py 实验")
    parser.add_argument("--config", help="JSON 配置文件（包含 experiments 数组）")
    parser.add_argument("--python", help="Python 可执行文件路径，默认使用 ./bd/bin/python")
    parser.add_argument("--project-root", help="项目根目录，默认脚本所在仓库根路径")
    parser.add_argument("--log-csv", help="运行记录 CSV 输出路径，默认 logs/phase3_runs.csv")
    parser.add_argument("--stop-on-failure", action="store_true", help="遇到错误立即退出")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令不执行")
    parser.add_argument("--extra-env", action="append", default=[], metavar="KEY=VALUE", help="额外设置的环境变量，可重复")
    parser.add_argument("--auto-summarize", action="store_true", help="所有实验完成后自动生成汇总表")
    parser.add_argument("--summary-pattern", dest="summary_patterns", action="append", help="汇总时使用的 glob 模式，可重复")
    parser.add_argument("--summary-output", dest="summary_output", help="汇总表写入 CSV 的路径")
    parser.add_argument("--summary-group-by", dest="summary_group_by", nargs="*", help="汇总后额外按列求平均")
    parser.add_argument("--summary-sort-by", dest="summary_sort_by", help="汇总排序字段，默认 mean_accuracy")
    parser.add_argument("--summary-desc", dest="summary_desc", action="store_true", help="汇总排序使用降序")
    parser.add_argument("--summary-include-empty", dest="summary_include_empty", action="store_true", help="汇总时包含缺失指标的目录")
    parser.add_argument("--cleanup-empty", action="store_true", help="在汇总前归档缺失或空指标的旧目录")
    parser.add_argument("--cleanup-pattern", dest="cleanup_patterns", action="append", help="匹配需要清理的目录模式，可重复")
    parser.add_argument("--cleanup-archive", dest="cleanup_archive", help="旧目录归档位置（默认 archive/old_results/时间戳）")

    # 单次实验可选参数
    parser.add_argument("--name", help="实验名称，便于记录")
    parser.add_argument("--notes", help="备注信息，写入 CSV/metadata")
    parser.add_argument("--type", dest="type")
    parser.add_argument("--evaluate-mode", dest="evaluate_mode")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--n-subject", dest="n_subject", type=int)
    parser.add_argument("--n-aug", dest="n_aug", type=int)
    parser.add_argument("--n-seg", dest="n_seg", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--depth", type=int)
    parser.add_argument("--emb-dim", dest="emb_dim", type=int)
    parser.add_argument("--validate-ratio", dest="validate_ratio", type=float)
    parser.add_argument("--run-tag", dest="run_tag")

    args = parser.parse_args()

    project_root = Path(args.project_root) if args.project_root else _project_root()
    python_exec = Path(args.python) if args.python else project_root / "bd" / "bin" / "python"
    log_csv = Path(args.log_csv) if args.log_csv else project_root / "logs" / "phase3_runs.csv"

    extra_env: Dict[str, str] = {}
    for item in args.extra_env:
        if "=" not in item:
            parser.error(f"--extra-env 参数格式应为 KEY=VALUE，实际为: {item}")
        key, value = item.split("=", 1)
        extra_env[key] = value

    experiments: List[Dict[str, Any]] = []
    if args.config:
        experiments.extend(_load_config(Path(args.config)))
    experiments.extend(_create_experiments_from_args(args))

    if not experiments:
        parser.error("未指定任何实验配置。使用 --config 或命令行参数。")

    for idx, cfg in enumerate(experiments, start=1):
        print(f"\n>>> 实验 {idx}/{len(experiments)}")
        run_experiment(cfg, python_exec, project_root, log_csv, extra_env, args.dry_run, args.stop_on_failure)

    if args.dry_run:
        return

    if args.cleanup_empty:
        cleanup_patterns = args.cleanup_patterns or ["CTNet_*"]
        print("\n=== 旧目录清理 ===")
        moved_items = cleanup_result_directories(project_root, cleanup_patterns, args.cleanup_archive)
        if moved_items:
            for item in moved_items:
                print(f"已归档 {item['original']} -> {item['archived']}")
        else:
            print("未发现需要归档的旧目录。")

    if (
        args.auto_summarize
        or args.summary_patterns
        or args.summary_output
        or args.summary_group_by
        or args.summary_sort_by
    ):
        patterns = args.summary_patterns or ["A_heads_*", "B_heads_*", "CTNet_*"]
        output_path = Path(args.summary_output) if args.summary_output else None
        sort_by = args.summary_sort_by or "mean_accuracy"
        _generate_summary_report(
            project_root,
            patterns,
            output_path,
            args.summary_group_by,
            sort_by,
            args.summary_desc,
            args.summary_include_empty,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[错误] {exc}")
        sys.exit(1)
