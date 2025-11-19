#!/usr/bin/env python3
"""汇总 CTNet Phase 3 运行结果并导出表格。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from phase3_utils import collect_result_records


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def summarize_results(
    root: Path,
    patterns: List[str],
    include_empty: bool = False,
    sort_by: str | None = "mean_accuracy",
    descending: bool = False,
    group_by: List[str] | None = None,
) -> pd.DataFrame:
    records = collect_result_records(root, patterns, include_empty=include_empty)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if sort_by and sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=not descending)
    if group_by:
        missing = [col for col in group_by if col not in df.columns]
        if missing:
            print(f"[警告] 以下分组列不存在：{missing}")
        else:
            agg_cols = [col for col in df.columns if col.startswith("mean_")]
            if agg_cols:
                grouped = (
                    df.groupby(group_by)[agg_cols]
                    .mean()
                    .reset_index()
                    .sort_values(sort_by, ascending=not descending, ignore_index=True)
                )
                print("\n=== 分组均值 ===")
                print(grouped.to_string(index=False))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 result_metric.xlsx 指标")
    parser.add_argument("--root", help="结果目录根路径，默认仓库根目录", default=None)
    parser.add_argument(
        "--pattern",
        action="append",
        default=["CTNet_*"],
        help="glob 模式（可重复），默认 CTNet_*",
    )
    parser.add_argument("--output", help="将结果写入 CSV 文件")
    parser.add_argument("--group-by", nargs="*", help="按照指定列进行聚合 (mean)")
    parser.add_argument("--sort-by", default="mean_accuracy", help="排序字段")
    parser.add_argument("--descending", action="store_true", help="降序排序")
    parser.add_argument(
        "--include-empty", action="store_true", help="保留缺失指标的目录"
    )

    args = parser.parse_args()
    root = Path(args.root) if args.root else _project_root()

    df = summarize_results(
        root,
        args.pattern,
        include_empty=args.include_empty,
        sort_by=args.sort_by,
        descending=args.descending,
        group_by=args.group_by,
    )

    if df.empty:
        print("未找到可用的结果目录。")
        return

    print("=== 结果汇总 ===")
    print(df.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n已写入 {output_path}")


if __name__ == "__main__":
    main()
