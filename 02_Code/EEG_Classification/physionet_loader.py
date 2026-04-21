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

# 事件编码 (2类模式)
EVENT_DICT = {
    'rest': 0,       # T0
    'left/hands': 1,  # T1
    'right/feet': 2,  # T2
}

# 4类标签映射
# left_right_imagine: T1=left, T2=right
# hands_feet_imagine: T1=hands(up), T2=feet(down)
LABEL_MAP_4CLASS = {
    'left': 0,    # 左手 → left
    'right': 1,   # 右手 → right
    'hands': 2,   # 双手 → up
    'feet': 3,    # 双脚 → down
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


def load_subject_4class_mne(
    subject: int,
    tmin: float = TMIN,
    tmax: float = TMAX,
    baseline: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载单个被试的 4 类数据（左手、右手、双手、双脚）
    
    合并 left_right_imagine 和 hands_feet_imagine 两种任务
    
    参数:
        subject: 被试编号 (1-109)
        tmin, tmax: epoch 时间范围
        baseline: 基线校正范围
    
    返回:
        data: (n_epochs, n_channels, n_times)
        labels: (n_epochs,) - 0=left, 1=right, 2=hands(up), 3=feet(down)
    """
    if not MNE_AVAILABLE:
        raise ImportError("需要安装 MNE: pip install mne")
    
    all_data = []
    all_labels = []
    
    # 加载 left_right_imagine (runs 4, 8, 12)
    # T1 = left (label 0), T2 = right (label 1)
    lr_runs = MI_RUNS['left_right_imagine']
    raw_fnames = eegbci.load_data(subject, lr_runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw_lr = mne.concatenate_raws(raws)
    eegbci.standardize(raw_lr)
    montage = mne.channels.make_standard_montage('standard_1005')
    raw_lr.set_montage(montage, on_missing='ignore')
    
    events_lr, event_id_lr = mne.events_from_annotations(raw_lr, verbose=False)
    picks = mne.pick_types(raw_lr.info, eeg=True, stim=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs_lr = mne.Epochs(
            raw_lr, events_lr, event_id_lr,
            tmin=tmin, tmax=tmax,
            proj=True, picks=picks,
            baseline=baseline,
            preload=True,
            verbose=False
        )
    
    data_lr = epochs_lr.get_data()
    labels_lr = epochs_lr.events[:, -1]
    
    # 将 MNE 事件 ID 映射到我们的标签
    # 通常 T1=1 → left(0), T2=2 → right(1), T0=0 → rest (过滤掉)
    unique_lr = np.unique(labels_lr)
    sorted_lr = sorted(unique_lr)
    # sorted_lr 通常是 [T0_id, T1_id, T2_id]
    # 我们只要 T1 和 T2
    if len(sorted_lr) >= 3:
        t0_id, t1_id, t2_id = sorted_lr[0], sorted_lr[1], sorted_lr[2]
    else:
        t1_id, t2_id = sorted_lr[0], sorted_lr[1]
        t0_id = -1
    
    # 过滤 rest 并映射标签
    mask_lr = labels_lr != t0_id
    data_lr = data_lr[mask_lr]
    labels_lr = labels_lr[mask_lr]
    labels_lr = np.where(labels_lr == t1_id, 0, 1)  # T1→left(0), T2→right(1)
    
    all_data.append(data_lr)
    all_labels.append(labels_lr)
    
    # 加载 hands_feet_imagine (runs 6, 10, 14)
    # T1 = hands (label 2), T2 = feet (label 3)
    hf_runs = MI_RUNS['hands_feet_imagine']
    raw_fnames = eegbci.load_data(subject, hf_runs)
    raws = [read_raw_edf(f, preload=True, verbose=False) for f in raw_fnames]
    raw_hf = mne.concatenate_raws(raws)
    eegbci.standardize(raw_hf)
    raw_hf.set_montage(montage, on_missing='ignore')
    
    events_hf, event_id_hf = mne.events_from_annotations(raw_hf, verbose=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        epochs_hf = mne.Epochs(
            raw_hf, events_hf, event_id_hf,
            tmin=tmin, tmax=tmax,
            proj=True, picks=picks,
            baseline=baseline,
            preload=True,
            verbose=False
        )
    
    data_hf = epochs_hf.get_data()
    labels_hf = epochs_hf.events[:, -1]
    
    unique_hf = np.unique(labels_hf)
    sorted_hf = sorted(unique_hf)
    if len(sorted_hf) >= 3:
        t0_id, t1_id, t2_id = sorted_hf[0], sorted_hf[1], sorted_hf[2]
    else:
        t1_id, t2_id = sorted_hf[0], sorted_hf[1]
        t0_id = -1
    
    mask_hf = labels_hf != t0_id
    data_hf = data_hf[mask_hf]
    labels_hf = labels_hf[mask_hf]
    labels_hf = np.where(labels_hf == t1_id, 2, 3)  # T1→hands(2), T2→feet(3)
    
    all_data.append(data_hf)
    all_labels.append(labels_hf)
    
    # 合并
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    return data.astype(np.float32), labels.astype(np.int64)


def load_multiple_subjects_4class_mne(
    subjects: List[int],
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载多个被试的 4 类数据
    
    返回:
        all_data: (total_epochs, n_channels, n_times)
        all_labels: (total_epochs,) - 0=left, 1=right, 2=hands, 3=feet
        subject_ids: (total_epochs,)
    """
    all_data = []
    all_labels = []
    subject_ids = []
    
    for subject in subjects:
        if verbose:
            print(f"  加载 Subject {subject:03d} (4类)...", end=" ")
        
        try:
            data, labels = load_subject_4class_mne(subject)
            all_data.append(data)
            all_labels.append(labels)
            subject_ids.append(np.full(len(labels), subject))
            
            # 统计各类数量
            n_left = (labels == 0).sum()
            n_right = (labels == 1).sum()
            n_hands = (labels == 2).sum()
            n_feet = (labels == 3).sum()
            
            if verbose:
                print(f"✓ {len(labels)} epochs (L:{n_left} R:{n_right} H:{n_hands} F:{n_feet})")
        except Exception as e:
            if verbose:
                print(f"✗ {e}")
    
    return (
        np.concatenate(all_data, axis=0),
        np.concatenate(all_labels, axis=0),
        np.concatenate(subject_ids, axis=0),
    )


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
    p.add_argument("--task", choices=['left_right', 'hands_feet', 'all', '4class'], 
                   default='left_right',
                   help="任务类型: left_right(2类), hands_feet(2类), 4class(4类合并)")
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
    
    # 4类模式特殊处理
    if args.task == '4class':
        print(f"\n配置:")
        print(f"  被试: {args.subjects}")
        print(f"  任务: 4class (左手/右手/双手/双脚)")
        print(f"  标签: 0=left, 1=right, 2=hands(up), 3=feet(down)")
        
        # 加载 4 类数据
        print(f"\n加载 4 类数据...")
        print("  (使用 MNE 自动下载，首次运行会较慢)")
        data, labels, subject_ids = load_multiple_subjects_4class_mne(
            args.subjects, verbose=True
        )
    else:
        # 确定要加载的 runs (2类模式)
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

