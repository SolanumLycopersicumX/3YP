#!/usr/bin/env python3
"""
PhysioNet EEG Motor Movement/Imagery Dataset 下载脚本

数据集: https://physionet.org/content/eegmmidb/1.0.0/
论文: Schalk et al., "BCI2000: A General-Purpose Brain-Computer Interface (BCI) System", 
      IEEE Trans. Biomed. Eng. 51(6):1034-1043, 2004

特点:
- 109 名被试
- 64 通道 (10-10 系统)
- 160 Hz 采样率
- 4 种运动/想象任务
- EDF+ 格式
- 总大小: ~3.4 GB (可按被试单独下载)

创建时间: 2026-02-10
"""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional
import ssl


# ============================================================================
# 配置
# ============================================================================

PHYSIONET_BASE_URL = "https://physionet.org/files/eegmmidb/1.0.0/"
DATASET_NAME = "eegmmidb"
VERSION = "1.0.0"

# 每个被试有 14 个 runs
RUNS_PER_SUBJECT = 14

# 任务说明
TASK_INFO = """
============================================================
PhysioNet EEG Motor Movement/Imagery Dataset
============================================================
数据集 URL: https://physionet.org/content/eegmmidb/1.0.0/
DOI: https://doi.org/10.13026/C28G6P

受试者: 109 名
通道数: 64 (10-10 系统)
采样率: 160 Hz
格式: EDF+

实验任务:
  Run 1:  基线 (睁眼)
  Run 2:  基线 (闭眼)
  Run 3:  任务1 - 实际打开/关闭左手或右手
  Run 4:  任务2 - 想象打开/关闭左手或右手
  Run 5:  任务3 - 实际打开/关闭双手或双脚
  Run 6:  任务4 - 想象打开/关闭双手或双脚
  Run 7-14: 重复任务1-4

标注:
  T0: 休息
  T1: 左手(runs 3,4,7,8,11,12) 或 双手(runs 5,6,9,10,13,14) 运动开始
  T2: 右手(runs 3,4,7,8,11,12) 或 双脚(runs 5,6,9,10,13,14) 运动开始
============================================================
"""


# ============================================================================
# 下载功能
# ============================================================================

def download_subject_wget(subject_id: int, output_dir: Path, verbose: bool = True):
    """使用 wget 下载单个被试数据"""
    
    subject_folder = f"S{subject_id:03d}"
    subject_url = f"{PHYSIONET_BASE_URL}{subject_folder}/"
    output_path = output_dir / subject_folder
    
    if output_path.exists() and any(output_path.iterdir()):
        if verbose:
            print(f"  ⏭ Subject {subject_id:03d}: 已存在，跳过")
        return True
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  📥 Subject {subject_id:03d}: 下载中...")
    
    try:
        # 使用 wget 下载整个被试目录
        cmd = [
            "wget", "-r", "-N", "-c", "-np", "-nH",
            "--cut-dirs=3",  # 去掉 files/eegmmidb/1.0.0 前缀
            "-P", str(output_dir),
            "-q" if not verbose else "-q",  # 静默模式
            subject_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            # 检查下载的文件
            edf_files = list(output_path.glob("*.edf"))
            if verbose:
                print(f"    ✓ 下载完成: {len(edf_files)} 个 EDF 文件")
            return True
        else:
            print(f"    ✗ wget 错误: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ✗ 下载超时")
        return False
    except Exception as e:
        print(f"    ✗ 下载失败: {e}")
        return False


def download_subject_urllib(subject_id: int, output_dir: Path, verbose: bool = True):
    """使用 urllib 下载单个被试数据"""
    
    subject_folder = f"S{subject_id:03d}"
    output_path = output_dir / subject_folder
    
    if output_path.exists() and len(list(output_path.glob("*.edf"))) >= RUNS_PER_SUBJECT:
        if verbose:
            print(f"  ⏭ Subject {subject_id:03d}: 已存在，跳过")
        return True
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"  📥 Subject {subject_id:03d}: 下载中...")
    
    # SSL 上下文
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    downloaded = 0
    
    for run in range(1, RUNS_PER_SUBJECT + 1):
        filename = f"{subject_folder}R{run:02d}.edf"
        url = f"{PHYSIONET_BASE_URL}{subject_folder}/{filename}"
        output_file = output_path / filename
        
        if output_file.exists():
            downloaded += 1
            continue
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, context=ctx, timeout=60) as response:
                with open(output_file, 'wb') as f:
                    f.write(response.read())
            downloaded += 1
        except Exception as e:
            if verbose:
                print(f"    ⚠ Run {run}: {e}")
    
    if verbose:
        print(f"    ✓ 下载完成: {downloaded}/{RUNS_PER_SUBJECT} 个 EDF 文件")
    
    return downloaded == RUNS_PER_SUBJECT


def download_subjects(
    subjects: List[int],
    output_dir: Path,
    method: str = "wget",
    verbose: bool = True
):
    """下载多个被试数据"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n下载 {len(subjects)} 个被试的数据到 {output_dir}/")
    print("-" * 50)
    
    success_count = 0
    
    for subject_id in subjects:
        if method == "wget":
            success = download_subject_wget(subject_id, output_dir, verbose)
        else:
            success = download_subject_urllib(subject_id, output_dir, verbose)
        
        if success:
            success_count += 1
    
    print("-" * 50)
    print(f"完成: {success_count}/{len(subjects)} 个被试下载成功")
    
    return success_count == len(subjects)


def estimate_download_size(subjects: List[int]) -> str:
    """估算下载大小"""
    # 每个被试约 31 MB
    size_per_subject_mb = 31
    total_mb = len(subjects) * size_per_subject_mb
    
    if total_mb < 1024:
        return f"{total_mb} MB"
    else:
        return f"{total_mb / 1024:.2f} GB"


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="PhysioNet EEGMMIDB 数据集下载",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=TASK_INFO
    )
    
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="要下载的被试编号 (1-109)，例如: --subjects 1 2 3")
    p.add_argument("--range", type=int, nargs=2, metavar=("START", "END"),
                   help="下载被试范围，例如: --range 1 10")
    p.add_argument("--all", action="store_true",
                   help="下载所有 109 个被试")
    p.add_argument("--output-dir", type=Path, 
                   default=Path("./physionet_raw/"),
                   help="输出目录 (默认: ./physionet_raw/)")
    p.add_argument("--method", choices=["wget", "urllib"], default="wget",
                   help="下载方法 (默认: wget)")
    p.add_argument("--info", action="store_true",
                   help="只显示数据集信息，不下载")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print(TASK_INFO)
    
    if args.info:
        return
    
    # 确定要下载的被试
    if args.all:
        subjects = list(range(1, 110))
    elif args.range:
        subjects = list(range(args.range[0], args.range[1] + 1))
    elif args.subjects:
        subjects = args.subjects
    else:
        # 默认下载前 3 个被试
        subjects = [1, 2, 3]
        print("未指定被试，默认下载前 3 个被试...")
    
    # 验证被试范围
    subjects = [s for s in subjects if 1 <= s <= 109]
    
    if not subjects:
        print("错误: 没有有效的被试编号 (范围: 1-109)")
        return
    
    # 估算大小
    size_estimate = estimate_download_size(subjects)
    print(f"\n📊 下载计划:")
    print(f"   被试数量: {len(subjects)}")
    print(f"   预计大小: {size_estimate}")
    print(f"   输出目录: {args.output_dir}")
    
    # 开始下载
    download_subjects(subjects, args.output_dir, args.method)
    
    print("\n✅ 下载完成!")
    print(f"   数据位置: {args.output_dir.absolute()}")
    print("\n下一步: 运行 physionet_loader.py 加载和预处理数据")


if __name__ == "__main__":
    main()

