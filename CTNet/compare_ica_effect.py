#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICA 去伪迹效果对比实验
对比使用ICA和不使用ICA预处理后的CTNet分类效果
"""

import os
import sys
import numpy as np
import scipy.io
from pathlib import Path
import json
from datetime import datetime

# MNE for preprocessing
try:
    import mne
    from mne.preprocessing import ICA
    from mne.channels import make_standard_montage
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("警告: MNE-Python 未安装，无法进行ICA预处理")
    print("请运行: pip install mne")

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "BCICIV_2a_gdf"  # 原始GDF数据
OUTPUT_DIR = SCRIPT_DIR / "ica_comparison"  # 输出目录

# 通道映射（BCICIV 2a数据集）
CHANNEL_RENAME_MAP = {
    "EEG-Fz": "Fz", "EEG-0": "FC3", "EEG-1": "FC1", "EEG-2": "FCz",
    "EEG-3": "FC2", "EEG-4": "FC4", "EEG-5": "C5", "EEG-C3": "C3",
    "EEG-6": "C1", "EEG-Cz": "Cz", "EEG-7": "C2", "EEG-C4": "C4",
    "EEG-8": "C6", "EEG-9": "CP3", "EEG-10": "CP1", "EEG-11": "CPz",
    "EEG-12": "CP2", "EEG-13": "CP4", "EEG-14": "P1", "EEG-Pz": "Pz",
    "EEG-15": "P2", "EEG-16": "POz",
}
EOG_CHANNELS = ["EOG-left", "EOG-central", "EOG-right"]

# 事件ID映射
EVENT_ID = {
    "left_hand": 1,
    "right_hand": 2,
    "foot": 3,
    "tongue": 4,
}


def load_gdf_raw(subject_file: str, preload: bool = True) -> mne.io.BaseRaw:
    """加载GDF文件"""
    gdf_path = DATA_ROOT / subject_file
    if not gdf_path.exists():
        raise FileNotFoundError(f"GDF文件不存在: {gdf_path}")
    raw = mne.io.read_raw_gdf(str(gdf_path), preload=preload, verbose="ERROR")
    return raw


def prepare_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """准备通道：重命名、设置EOG类型、设置蒙太奇"""
    raw = raw.copy()
    
    # 重命名EEG通道
    available_map = {old: new for old, new in CHANNEL_RENAME_MAP.items() if old in raw.ch_names}
    if available_map:
        raw.rename_channels(available_map)
    
    # 设置EOG通道类型
    channel_types = {ch: "eog" for ch in EOG_CHANNELS if ch in raw.ch_names}
    if channel_types:
        raw.set_channel_types(channel_types, verbose="ERROR")
    
    # 设置蒙太奇
    try:
        montage = make_standard_montage("standard_1020")
        raw.set_montage(montage, match_case=False, on_missing="warn", verbose="ERROR")
    except Exception as e:
        print(f"  警告: 设置蒙太奇失败 ({e})")
    
    return raw


def bandpass_filter(raw: mne.io.BaseRaw, l_freq: float = 8.0, h_freq: float = 30.0) -> mne.io.BaseRaw:
    """带通滤波 8-30Hz"""
    raw_filt = raw.copy()
    if not raw_filt.preload:
        raw_filt.load_data()
    raw_filt.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", 
                    method="fir", phase="zero-double", verbose="ERROR")
    return raw_filt


def run_ica_cleaning(raw: mne.io.BaseRaw, n_components: int = 20) -> mne.io.BaseRaw:
    """运行ICA并去除眼电伪迹"""
    # 拟合ICA
    eeg_count = len(raw.copy().pick_types(eeg=True, exclude=[]).ch_names)
    n_components = min(n_components, eeg_count)
    
    ica = ICA(n_components=n_components, method="fastica", random_state=97, 
              max_iter="auto", verbose="ERROR")
    ica.fit(raw, decim=3, verbose="ERROR")
    
    # 检测EOG相关成分
    bad_inds = set()
    for ch in EOG_CHANNELS:
        if ch in raw.ch_names:
            try:
                inds, _ = ica.find_bads_eog(raw, ch_name=ch, threshold=3.0, verbose="ERROR")
                bad_inds.update(int(i) for i in inds)
            except Exception:
                pass
    
    # 应用ICA去除伪迹
    if bad_inds:
        raw_clean = raw.copy()
        ica_copy = ica.copy()
        ica_copy.exclude = list(bad_inds)
        ica_copy.apply(raw_clean, verbose="ERROR")
        print(f"    ICA去除了 {len(bad_inds)} 个伪迹成分: {sorted(bad_inds)}")
        return raw_clean
    else:
        print("    ICA未检测到明显伪迹成分")
        return raw.copy()


def create_epochs(raw: mne.io.BaseRaw, tmin: float = 0.0, tmax: float = 4.0) -> mne.Epochs:
    """创建Epochs"""
    events, events_map = mne.events_from_annotations(raw, verbose="ERROR")
    
    # 映射事件ID
    cue_map = {"left_hand": "769", "right_hand": "770", "foot": "771", "tongue": "772"}
    event_id = {}
    for label, code in cue_map.items():
        if code in events_map:
            event_id[label] = events_map[code]
    
    if not event_id:
        raise RuntimeError("未找到有效的事件码")
    
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, picks="eeg", preload=True, verbose="ERROR")
    return epochs


def epochs_to_mat(epochs: mne.Epochs, output_path: Path, label_mapping: dict = None):
    """将Epochs转换为CTNet使用的mat格式"""
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    
    # 获取标签
    if label_mapping is None:
        label_mapping = {"left_hand": 1, "right_hand": 2, "foot": 3, "tongue": 4}
    
    labels = []
    for event in epochs.events:
        event_code = event[2]
        for label_name, code in epochs.event_id.items():
            if code == event_code:
                labels.append(label_mapping.get(label_name, 0))
                break
    labels = np.array(labels).reshape(-1, 1)
    
    # 保存为mat格式
    scipy.io.savemat(str(output_path), {"data": data, "label": labels})
    print(f"    保存: {output_path} | 形状: {data.shape} | 标签: {len(labels)}")


def process_subject(subject_id: int, use_ica: bool = True):
    """处理单个受试者的数据"""
    train_file = f"A{subject_id:02d}T.gdf"
    test_file = f"A{subject_id:02d}E.gdf"
    
    print(f"\n处理受试者 {subject_id} ({'使用ICA' if use_ica else '不使用ICA'})...")
    
    results = {"train": None, "test": None}
    
    for mode, gdf_file in [("train", train_file), ("test", test_file)]:
        try:
            # 加载数据
            raw = load_gdf_raw(gdf_file)
            print(f"  {mode}: 加载 {gdf_file}")
            
            # 准备通道
            raw = prepare_channels(raw)
            
            # 带通滤波
            raw = bandpass_filter(raw)
            print(f"    带通滤波 8-30Hz 完成")
            
            # ICA去伪迹（可选）
            if use_ica:
                raw = run_ica_cleaning(raw)
            
            # 创建Epochs
            epochs = create_epochs(raw)
            print(f"    创建Epochs: {len(epochs)} 个试次")
            
            results[mode] = epochs
            
        except Exception as e:
            print(f"  错误: {e}")
            return None
    
    return results


def save_processed_data(all_results: dict, output_subdir: str):
    """保存所有处理后的数据"""
    output_path = OUTPUT_DIR / output_subdir
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存数据到: {output_path}")
    
    for subject_id, results in all_results.items():
        if results is None:
            continue
        
        train_epochs = results["train"]
        test_epochs = results["test"]
        
        if train_epochs is not None:
            train_mat = output_path / f"A{subject_id:02d}T.mat"
            epochs_to_mat(train_epochs, train_mat)
        
        if test_epochs is not None:
            test_mat = output_path / f"A{subject_id:02d}E.mat"
            epochs_to_mat(test_epochs, test_mat)
    
    return output_path


def run_ctnet_training(data_dir: Path, result_name: str, epochs: int = 500):
    """运行CTNet训练"""
    print(f"\n{'='*70}")
    print(f"开始CTNet训练: {result_name}")
    print(f"数据目录: {data_dir}")
    print(f"{'='*70}")
    
    # 导入CTNet模块
    sys.path.insert(0, str(SCRIPT_DIR))
    
    from CTNet_model import main as ctnet_main
    
    # 修改全局变量
    import CTNet_model
    original_data_dir = CTNet_model.DATA_DIR
    original_epochs = CTNet_model.EPOCHS
    
    CTNet_model.DATA_DIR = str(data_dir) + "/"
    CTNet_model.EPOCHS = epochs
    
    try:
        result = ctnet_main(
            result_name,
            evaluate_mode='subject-dependent',
            heads=2,
            emb_size=40,
            depth=6,
            dataset_type='A',
            validate_ratio=0.2,
            early_stopping=True,
            patience=50,
            min_delta=0.0001,
            verbose=True,
            plot_training=False,
            batch_size=72,
            learning_rate=0.001,
        )
        return result
    finally:
        # 恢复原始值
        CTNet_model.DATA_DIR = original_data_dir
        CTNet_model.EPOCHS = original_epochs


def compare_results(result_with_ica: dict, result_without_ica: dict):
    """对比两组结果"""
    print("\n" + "="*70)
    print("ICA效果对比结果")
    print("="*70)
    
    if result_with_ica is None or result_without_ica is None:
        print("错误: 缺少训练结果，无法对比")
        return
    
    # 提取指标
    df_with_ica = result_with_ica.get('result_metric')
    df_without_ica = result_without_ica.get('result_metric')
    
    if df_with_ica is None or df_without_ica is None:
        print("错误: 无法提取指标数据")
        return
    
    print(f"\n{'Subject':<10} {'无ICA Acc':<12} {'有ICA Acc':<12} {'差异':<10}")
    print("-" * 50)
    
    improvements = []
    for i in range(min(len(df_with_ica), len(df_without_ica))):
        acc_without = df_without_ica.iloc[i]['accuray']
        acc_with = df_with_ica.iloc[i]['accuray']
        diff = acc_with - acc_without
        improvements.append(diff)
        
        sign = "+" if diff > 0 else ""
        print(f"Subject {i+1:<3} {acc_without:>10.2f}%  {acc_with:>10.2f}%  {sign}{diff:>8.2f}%")
    
    print("-" * 50)
    
    avg_without = df_without_ica['accuray'].mean()
    avg_with = df_with_ica['accuray'].mean()
    avg_diff = avg_with - avg_without
    sign = "+" if avg_diff > 0 else ""
    
    print(f"{'平均':<10} {avg_without:>10.2f}%  {avg_with:>10.2f}%  {sign}{avg_diff:>8.2f}%")
    print("="*70)
    
    # 统计分析
    print(f"\n统计分析:")
    print(f"  - 使用ICA后准确率提升的受试者: {sum(1 for d in improvements if d > 0)}/9")
    print(f"  - 使用ICA后准确率下降的受试者: {sum(1 for d in improvements if d < 0)}/9")
    print(f"  - 平均提升: {np.mean(improvements):.2f}%")
    print(f"  - 最大提升: {max(improvements):.2f}%")
    print(f"  - 最大下降: {min(improvements):.2f}%")
    
    # 保存对比结果
    comparison_result = {
        "timestamp": datetime.now().isoformat(),
        "avg_accuracy_without_ica": float(avg_without),
        "avg_accuracy_with_ica": float(avg_with),
        "improvement": float(avg_diff),
        "per_subject_improvement": [float(d) for d in improvements],
    }
    
    result_file = OUTPUT_DIR / "comparison_result.json"
    with open(result_file, "w") as f:
        json.dump(comparison_result, f, indent=2)
    print(f"\n对比结果已保存: {result_file}")


def main():
    """主函数"""
    print("="*70)
    print("ICA 去伪迹效果对比实验")
    print("="*70)
    
    if not MNE_AVAILABLE:
        print("错误: 需要安装MNE-Python")
        print("运行: pip install mne")
        return
    
    # 检查数据目录
    if not DATA_ROOT.exists():
        print(f"错误: 原始GDF数据目录不存在: {DATA_ROOT}")
        print("请确保BCICIV_2a_gdf目录存在并包含原始GDF文件")
        return
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 处理所有受试者
    print("\n" + "="*70)
    print("步骤1: 预处理数据（不使用ICA）")
    print("="*70)
    
    results_no_ica = {}
    for subject_id in range(1, 10):
        results_no_ica[subject_id] = process_subject(subject_id, use_ica=False)
    
    data_dir_no_ica = save_processed_data(results_no_ica, "data_no_ica")
    
    print("\n" + "="*70)
    print("步骤2: 预处理数据（使用ICA）")
    print("="*70)
    
    results_with_ica = {}
    for subject_id in range(1, 10):
        results_with_ica[subject_id] = process_subject(subject_id, use_ica=True)
    
    data_dir_with_ica = save_processed_data(results_with_ica, "data_with_ica")
    
    # 训练CTNet
    print("\n" + "="*70)
    print("步骤3: 训练CTNet（不使用ICA数据）")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_no_ica = run_ctnet_training(
        data_dir_no_ica, 
        f"CTNet_no_ica_{timestamp}",
        epochs=500
    )
    
    print("\n" + "="*70)
    print("步骤4: 训练CTNet（使用ICA数据）")
    print("="*70)
    
    result_with_ica = run_ctnet_training(
        data_dir_with_ica,
        f"CTNet_with_ica_{timestamp}",
        epochs=500
    )
    
    # 对比结果
    compare_results(result_with_ica, result_no_ica)


if __name__ == "__main__":
    main()



