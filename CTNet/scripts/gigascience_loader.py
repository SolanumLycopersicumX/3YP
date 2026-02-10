#!/usr/bin/env python3
"""
GigaScience 数据集加载器

数据集信息:
- 论文: Jeong et al., "Multimodal signal dataset for 11 intuitive movement tasks", GigaScience, 2020
- 通道: 64 EEG
- 类别: 11 种直觉运动任务
- 采样率: 1000 Hz
- 被试: 25 人

下载链接: http://gigadb.org/dataset/100788

任务类别:
1. Supination (前臂旋后)
2. Pronation (前臂旋前)
3. Extension (腕部伸展)
4. Flexion (腕部屈曲)
5. Extension of all fingers (全部手指伸展)
6. Fist (握拳)
7. Lateral grasp (侧向抓握)
8. Palmar grasp (掌心抓握)
9. Elbow flexion (肘部屈曲)
10. Elbow extension (肘部伸展)
11. Rest (休息)

创建时间: 2026-02-10
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# GigaScience 数据集配置
# ============================================================================

@dataclass
class GigaScienceConfig:
    """GigaScience 数据集配置"""
    n_subjects: int = 25
    n_channels: int = 64
    n_classes: int = 11
    original_sfreq: int = 1000  # Hz
    target_sfreq: int = 250     # Hz (与 BCI Competition IV 一致)
    trial_duration: float = 4.0  # seconds
    
    # 类别名称
    class_names: List[str] = None
    
    def __post_init__(self):
        self.class_names = [
            "Supination",      # 1
            "Pronation",       # 2
            "Wrist Extension", # 3
            "Wrist Flexion",   # 4
            "Finger Extension",# 5
            "Fist",            # 6
            "Lateral Grasp",   # 7
            "Palmar Grasp",    # 8
            "Elbow Flexion",   # 9
            "Elbow Extension", # 10
            "Rest",            # 11
        ]


# ============================================================================
# 数据加载函数
# ============================================================================

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, 
                    fs: int, order: int = 4) -> np.ndarray:
    """带通滤波"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=-1)


def load_gigascience_subject(
    data_dir: Path,
    subject: int,
    config: Optional[GigaScienceConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载单个被试的 GigaScience 数据
    
    Parameters
    ----------
    data_dir : Path
        数据目录路径
    subject : int
        被试编号 (1-25)
    config : GigaScienceConfig
        数据集配置
    
    Returns
    -------
    data : np.ndarray
        形状 (n_trials, n_channels, n_samples)
    labels : np.ndarray
        形状 (n_trials,)
    """
    if config is None:
        config = GigaScienceConfig()
    
    # 尝试多种可能的文件命名方式
    possible_patterns = [
        f"sub{subject:02d}.mat",
        f"subject{subject:02d}.mat", 
        f"S{subject:02d}.mat",
        f"s{subject:02d}.mat",
        f"sub_{subject:02d}.mat",
    ]
    
    mat_file = None
    for pattern in possible_patterns:
        path = data_dir / pattern
        if path.exists():
            mat_file = path
            break
    
    if mat_file is None:
        raise FileNotFoundError(
            f"找不到被试 {subject} 的数据文件。"
            f"尝试的模式: {possible_patterns}"
        )
    
    # 加载 .mat 文件
    mat_data = sio.loadmat(str(mat_file))
    
    # 尝试不同的键名
    data_keys = ['data', 'eeg', 'EEG', 'X', 'x', 'signal']
    label_keys = ['label', 'labels', 'y', 'Y', 'class', 'classes']
    
    data = None
    labels = None
    
    for key in data_keys:
        if key in mat_data:
            data = mat_data[key]
            break
    
    for key in label_keys:
        if key in mat_data:
            labels = mat_data[key]
            break
    
    if data is None:
        available_keys = [k for k in mat_data.keys() if not k.startswith('_')]
        raise KeyError(f"找不到数据键。可用键: {available_keys}")
    
    if labels is None:
        available_keys = [k for k in mat_data.keys() if not k.startswith('_')]
        raise KeyError(f"找不到标签键。可用键: {available_keys}")
    
    # 确保标签是一维的
    labels = labels.flatten()
    
    return data, labels


def preprocess_gigascience(
    data: np.ndarray,
    labels: np.ndarray,
    config: Optional[GigaScienceConfig] = None,
    lowcut: float = 8.0,
    highcut: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    预处理 GigaScience 数据
    
    1. 带通滤波 (8-30 Hz, 运动想象频段)
    2. 降采样到 250 Hz
    3. 归一化
    
    Parameters
    ----------
    data : np.ndarray
        原始数据 (n_trials, n_channels, n_samples)
    labels : np.ndarray
        标签
    config : GigaScienceConfig
        配置
    
    Returns
    -------
    processed_data : np.ndarray
    labels : np.ndarray
    """
    if config is None:
        config = GigaScienceConfig()
    
    n_trials, n_channels, n_samples = data.shape
    
    # 1. 带通滤波
    print(f"  带通滤波 ({lowcut}-{highcut} Hz)...")
    data = bandpass_filter(data, lowcut, highcut, config.original_sfreq)
    
    # 2. 降采样
    if config.target_sfreq != config.original_sfreq:
        ratio = config.target_sfreq / config.original_sfreq
        new_n_samples = int(n_samples * ratio)
        print(f"  降采样 {config.original_sfreq}Hz → {config.target_sfreq}Hz ({n_samples} → {new_n_samples} samples)...")
        
        resampled = np.zeros((n_trials, n_channels, new_n_samples))
        for i in range(n_trials):
            for j in range(n_channels):
                resampled[i, j] = resample(data[i, j], new_n_samples)
        data = resampled
    
    # 3. Z-score 归一化
    print("  Z-score 归一化...")
    mean = data.mean()
    std = data.std()
    data = (data - mean) / (std + 1e-8)
    
    return data, labels


def convert_to_mat_format(
    data: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    subject: int,
    mode: str = "train",
):
    """
    将数据转换为与 BCI Competition IV 相同的 .mat 格式
    
    格式: G{subject:02d}{T/E}.mat
    内容: data (n_trials, n_channels, n_samples), label (n_trials, 1)
    """
    suffix = "T" if mode == "train" else "E"
    filename = f"G{subject:02d}{suffix}.mat"
    filepath = output_path / filename
    
    sio.savemat(str(filepath), {
        'data': data,
        'label': labels.reshape(-1, 1),
    })
    
    print(f"  保存: {filepath}")
    return filepath


# ============================================================================
# 数据集集成
# ============================================================================

def integrate_gigascience(
    raw_data_dir: Path,
    output_dir: Path,
    subjects: Optional[List[int]] = None,
    train_ratio: float = 0.8,
    config: Optional[GigaScienceConfig] = None,
) -> Dict[str, any]:
    """
    集成 GigaScience 数据集到项目格式
    
    Parameters
    ----------
    raw_data_dir : Path
        原始 GigaScience 数据目录
    output_dir : Path
        输出目录 (与 mymat_raw 同级或相同)
    subjects : List[int]
        要处理的被试列表 (默认全部 1-25)
    train_ratio : float
        训练集比例
    config : GigaScienceConfig
        配置
    
    Returns
    -------
    summary : Dict
    """
    if config is None:
        config = GigaScienceConfig()
    
    if subjects is None:
        subjects = list(range(1, config.n_subjects + 1))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary = {
        'subjects_processed': [],
        'subjects_failed': [],
        'total_trials': 0,
    }
    
    print("="*60)
    print("GigaScience 数据集集成")
    print("="*60)
    print(f"原始数据目录: {raw_data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"被试数量: {len(subjects)}")
    print("="*60)
    
    for subject in subjects:
        print(f"\n[Subject {subject}/{len(subjects)}]")
        
        try:
            # 加载数据
            print(f"  加载数据...")
            data, labels = load_gigascience_subject(raw_data_dir, subject, config)
            print(f"  原始形状: {data.shape}, 标签: {labels.shape}")
            
            # 预处理
            data, labels = preprocess_gigascience(data, labels, config)
            print(f"  处理后形状: {data.shape}")
            
            # 划分训练/测试集
            n_trials = len(labels)
            n_train = int(n_trials * train_ratio)
            
            # 随机打乱
            indices = np.random.permutation(n_trials)
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            train_data = data[train_idx]
            train_labels = labels[train_idx]
            test_data = data[test_idx]
            test_labels = labels[test_idx]
            
            print(f"  训练集: {len(train_labels)} trials, 测试集: {len(test_labels)} trials")
            
            # 保存
            convert_to_mat_format(train_data, train_labels, output_dir, subject, "train")
            convert_to_mat_format(test_data, test_labels, output_dir, subject, "test")
            
            summary['subjects_processed'].append(subject)
            summary['total_trials'] += n_trials
            
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            summary['subjects_failed'].append({'subject': subject, 'error': str(e)})
    
    print("\n" + "="*60)
    print("集成完成!")
    print(f"成功: {len(summary['subjects_processed'])} 被试")
    print(f"失败: {len(summary['subjects_failed'])} 被试")
    print(f"总试次: {summary['total_trials']}")
    print("="*60)
    
    return summary


# ============================================================================
# 扩展 utils.py 的加载函数
# ============================================================================

def number_class_channel_extended(dataset_type: str) -> Tuple[int, int]:
    """
    扩展的数据集配置函数，支持 GigaScience
    """
    if dataset_type == "A":
        return 4, 22   # IV-2a
    elif dataset_type == "B":
        return 2, 3    # IV-2b
    elif dataset_type == "G":
        return 11, 64  # GigaScience
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}")


def load_gigascience_evaluate(
    dir_path: Path,
    subject: int,
    evaluate_mode: str = "subject-dependent",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 GigaScience 数据 (与 load_data_evaluate 接口兼容)
    """
    train_path = dir_path / f"G{subject:02d}T.mat"
    test_path = dir_path / f"G{subject:02d}E.mat"
    
    if not train_path.exists():
        raise FileNotFoundError(f"找不到训练文件: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"找不到测试文件: {test_path}")
    
    train_mat = sio.loadmat(str(train_path))
    test_mat = sio.loadmat(str(test_path))
    
    return (
        train_mat['data'],
        train_mat['label'],
        test_mat['data'],
        test_mat['label'],
    )


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="GigaScience 数据集集成")
    
    p.add_argument("--raw-dir", type=Path, required=True,
                   help="原始 GigaScience 数据目录")
    p.add_argument("--output-dir", type=Path, default=Path("./mymat_raw/"),
                   help="输出目录")
    p.add_argument("--subjects", type=int, nargs="+", default=None,
                   help="要处理的被试列表 (默认全部)")
    p.add_argument("--train-ratio", type=float, default=0.8,
                   help="训练集比例")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    config = GigaScienceConfig()
    
    summary = integrate_gigascience(
        raw_data_dir=args.raw_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
        train_ratio=args.train_ratio,
        config=config,
    )
    
    # 保存摘要
    import json
    summary_path = args.output_dir / "gigascience_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n摘要已保存: {summary_path}")


if __name__ == "__main__":
    main()

