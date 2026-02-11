#!/usr/bin/env python3
"""
PhysioNet EEG Motor Movement/Imagery Dataset 加载器

数据集: https://physionet.org/content/eegmmidb/1.0.0/
使用 MNE 库原生加载 EDF 文件

特点:
- 109 名被试, 64 通道, 160 Hz
- 4 种运动/想象任务
- 支持导出为与 CTNet 兼容的 .mat 格式

创建时间: 2026-02-10
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# 尝试导入 MNE
try:
    import mne
    from mne.io import read_raw_edf
    from mne.datasets import eegbci
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("⚠️ MNE 未安装。请运行: pip install mne")

try:
    import scipy.io as sio
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

# 采样率
SFREQ = 160  # Hz

# 通道数
N_CHANNELS = 64

# 任务定义 (runs that contain motor imagery)
MI_RUNS = {
    'left_right_real': [3, 7, 11],      # 实际左右手
    'left_right_imagine': [4, 8, 12],   # 想象左右手
    'hands_feet_real': [5, 9, 13],      # 实际双手/双脚
    'hands_feet_imagine': [6, 10, 14],  # 想象双手/双脚
}

# 事件编码
EVENT_DICT = {
    'rest': 0,       # T0
    'left/hands': 1,  # T1
    'right/feet': 2,  # T2
}

# Epoch 参数
TMIN = -0.5  # 事件前 0.5 秒
TMAX = 4.0   # 事件后 4 秒 (试验持续约 4 秒)


# ============================================================================
# MNE 原生加载 (推荐)
# ============================================================================

def load_subject_mne(
    subject: int,
    runs: List[int],
    tmin: float = TMIN,
    tmax: float = TMAX,
    baseline: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 MNE 加载单个被试的数据
    
    参数:
        subject: 被试编号 (1-109)
        runs: 要加载的 runs (3-14 包含任务)
        tmin, tmax: epoch 时间范围
        baseline: 基线校正范围
    
    返回:
        data: (n_epochs, n_channels, n_times)
        labels: (n_epochs,) - 0=rest, 1=left/hands, 2=right/feet
    """
    if not MNE_AVAILABLE:
        raise ImportError("需要安装 MNE: pip install mne")
    
    # 使用 MNE 的 eegbci 模块自动下载和加载
    # 这是 MNE 推荐的方式，会自动缓存数据
    raw_fnames = eegbci.load_data(subject, runs)
    
    # 合并所有 runs
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw = mne.concatenate_raws(raws)
    
    # 标准化通道名称
    eegbci.standardize(raw)
    
    # 设置 montage
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    
    # 获取事件
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # 创建 epochs
    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=tmin, tmax=tmax,
            proj=True, picks=picks,
            baseline=baseline,
            preload=True,
            verbose=False
        )
    
    # 获取数据和标签
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    # 转换标签: T1=1, T2=2, T0=0
    labels = epochs.events[:, -1]
    
    # 标签映射 (MNE 的事件 ID 可能不同)
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels = np.array([label_map[l] for l in labels])
    
    return data.astype(np.float32), labels.astype(np.int64)


def load_multiple_subjects_mne(
    subjects: List[int],
    runs: List[int],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载多个被试的数据
    
    返回:
        all_data: (total_epochs, n_channels, n_times)
        all_labels: (total_epochs,)
        subject_ids: (total_epochs,) - 每个 epoch 属于哪个被试
    """
    all_data = []
    all_labels = []
    subject_ids = []
    
    for subject in subjects:
        if verbose:
            print(f"  加载 Subject {subject:03d}...", end=" ")
        
        try:
            data, labels = load_subject_mne(subject, runs)
            all_data.append(data)
            all_labels.append(labels)
            subject_ids.append(np.full(len(labels), subject))
            
            if verbose:
                print(f"✓ {len(labels)} epochs")
        except Exception as e:
            if verbose:
                print(f"✗ {e}")
    
    return (
        np.concatenate(all_data, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(subject_ids, axis=0)
    )


# ============================================================================
# 从本地 EDF 文件加载
# ============================================================================

def load_subject_local(
    subject: int,
    data_dir: Path,
    runs: List[int],
    tmin: float = TMIN,
    tmax: float = TMAX,
) -> Tuple[np.ndarray, np.ndarray]:
    """从本地下载的 EDF 文件加载"""
    
    if not MNE_AVAILABLE:
        raise ImportError("需要安装 MNE: pip install mne")
    
    subject_dir = data_dir / f"S{subject:03d}"
    
    if not subject_dir.exists():
        raise FileNotFoundError(f"被试目录不存在: {subject_dir}")
    
    raws = []
    for run in runs:
        edf_file = subject_dir / f"S{subject:03d}R{run:02d}.edf"
        if edf_file.exists():
            raw = read_raw_edf(edf_file, preload=True, verbose=False)
            raws.append(raw)
    
    if not raws:
        raise FileNotFoundError(f"未找到 EDF 文件: {subject_dir}")
    
    raw = mne.concatenate_raws(raws)
    
    # 获取事件
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    # 创建 epochs
    picks = mne.pick_types(raw.info, eeg=True, stim=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs = mne.Epochs(
            raw, events, event_id,
            tmin=tmin, tmax=tmax,
            proj=True, picks=picks,
            preload=True,
            verbose=False
        )
    
    data = epochs.get_data()
    labels = epochs.events[:, -1]
    
    # 标签映射
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(sorted(unique_labels))}
    labels = np.array([label_map[l] for l in labels])
    
    return data.astype(np.float32), labels.astype(np.int64)


# ============================================================================
# 导出为 CTNet 兼容格式
# ============================================================================

def export_to_mat(
    data: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    subject_ids: Optional[np.ndarray] = None,
):
    """
    导出为 .mat 格式，与 CTNet 的 load_data 兼容
    
    CTNet 期望的格式:
        data: (n_trials, n_channels, n_times)
        label: (n_trials,) 或 (n_trials, 1)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("需要安装 scipy: pip install scipy")
    
    mat_dict = {
        'data': data,
        'label': labels.reshape(-1, 1),  # CTNet 期望 (n, 1)
    }
    
    if subject_ids is not None:
        mat_dict['subject_id'] = subject_ids.reshape(-1, 1)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(str(output_path), mat_dict)
    
    print(f"✓ 导出到 {output_path}")
    print(f"  - data shape: {data.shape}")
    print(f"  - labels: {np.unique(labels)}")


# ============================================================================
# 预处理
# ============================================================================

def preprocess_data(
    data: np.ndarray,
    sfreq: float = SFREQ,
    l_freq: float = 8.0,
    h_freq: float = 30.0,
) -> np.ndarray:
    """
    基本预处理: 带通滤波
    
    参数:
        data: (n_epochs, n_channels, n_times)
        sfreq: 采样率
        l_freq, h_freq: 滤波频率范围
    """
    if not MNE_AVAILABLE:
        return data
    
    # MNE 需要 float64
    data_f64 = data.astype(np.float64)
    
    # 滤波
    filtered_data = mne.filter.filter_data(
        data_f64, sfreq, l_freq, h_freq,
        verbose=False
    )
    
    return filtered_data.astype(np.float32)


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PhysioNet EEGMMIDB 数据加载器")
    
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3],
                   help="被试编号")
    p.add_argument("--task", choices=['left_right', 'hands_feet', 'all'], 
                   default='left_right',
                   help="任务类型")
    p.add_argument("--imagine-only", action="store_true",
                   help="只加载运动想象任务 (不包括实际运动)")
    p.add_argument("--output-mat", type=Path, default=None,
                   help="导出为 .mat 文件")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="本地数据目录 (如果已下载)")
    p.add_argument("--filter", action="store_true",
                   help="应用 8-30 Hz 带通滤波")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("PhysioNet EEGMMIDB 数据加载器")
    print("="*60)
    
    if not MNE_AVAILABLE:
        print("\n❌ 需要安装 MNE 库:")
        print("   pip install mne")
        return
    
    # 确定要加载的 runs
    if args.task == 'left_right':
        if args.imagine_only:
            runs = MI_RUNS['left_right_imagine']
        else:
            runs = MI_RUNS['left_right_real'] + MI_RUNS['left_right_imagine']
    elif args.task == 'hands_feet':
        if args.imagine_only:
            runs = MI_RUNS['hands_feet_imagine']
        else:
            runs = MI_RUNS['hands_feet_real'] + MI_RUNS['hands_feet_imagine']
    else:  # all
        runs = list(range(3, 15))  # runs 3-14
    
    runs = sorted(set(runs))
    
    print(f"\n配置:")
    print(f"  被试: {args.subjects}")
    print(f"  任务: {args.task}")
    print(f"  Runs: {runs}")
    print(f"  想象任务only: {args.imagine_only}")
    
    # 加载数据
    print(f"\n加载数据...")
    
    if args.data_dir and args.data_dir.exists():
        # 从本地加载
        all_data = []
        all_labels = []
        all_subject_ids = []
        
        for subject in args.subjects:
            print(f"  Subject {subject:03d}...", end=" ")
            try:
                data, labels = load_subject_local(subject, args.data_dir, runs)
                all_data.append(data)
                all_labels.append(labels)
                all_subject_ids.append(np.full(len(labels), subject))
                print(f"✓ {len(labels)} epochs")
            except Exception as e:
                print(f"✗ {e}")
        
        data = np.concatenate(all_data, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        subject_ids = np.concatenate(all_subject_ids, axis=0)
    else:
        # 使用 MNE 自动下载
        print("  (使用 MNE 自动下载，首次运行会较慢)")
        data, labels, subject_ids = load_multiple_subjects_mne(
            args.subjects, runs, verbose=True
        )
    
    print(f"\n数据加载完成:")
    print(f"  Shape: {data.shape}")
    print(f"  标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 滤波
    if args.filter:
        print(f"\n应用 8-30 Hz 带通滤波...")
        data = preprocess_data(data)
        print("  ✓ 滤波完成")
    
    # 导出
    if args.output_mat:
        print(f"\n导出到 {args.output_mat}...")
        export_to_mat(data, labels, args.output_mat, subject_ids)
    
    print("\n✅ 完成!")
    
    return data, labels, subject_ids


if __name__ == "__main__":
    main()

