#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ICA 去伪迹效果对比实验（简化版）
使用sklearn的FastICA直接对mat数据进行处理
"""

import os
import sys
import warnings
# 过滤FastICA收敛警告
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.decomposition._fastica')

import numpy as np
import scipy.io
from scipy import signal
from pathlib import Path
import json
from datetime import datetime
from sklearn.decomposition import FastICA
import torch

# 路径配置
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "mymat_raw"  # 原始mat数据
OUTPUT_DIR = SCRIPT_DIR / "ica_comparison"

# 采样率（BCI Competition IV 2a）
SFREQ = 250  # Hz


def load_mat_data(subject_id: int, mode: str = 'train'):
    """加载mat数据"""
    mode_suffix = 'T' if mode == 'train' else 'E'
    mat_file = DATA_DIR / f"A{subject_id:02d}{mode_suffix}.mat"
    
    if not mat_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {mat_file}")
    
    data = scipy.io.loadmat(str(mat_file))
    X = data['data']  # (n_trials, n_channels, n_times)
    y = data['label'].flatten()
    
    return X, y


def bandpass_filter(data: np.ndarray, l_freq: float = 8.0, h_freq: float = 30.0, 
                    sfreq: float = SFREQ) -> np.ndarray:
    """带通滤波 8-30Hz"""
    nyq = sfreq / 2
    low = l_freq / nyq
    high = h_freq / nyq
    
    # 设计Butterworth带通滤波器
    b, a = signal.butter(4, [low, high], btype='band')
    
    # 对每个trial的每个通道进行滤波
    filtered_data = np.zeros_like(data)
    for i in range(data.shape[0]):  # trials
        for j in range(data.shape[1]):  # channels
            filtered_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    
    return filtered_data


def apply_ica_cleaning(data: np.ndarray, n_components: int = 20, 
                       n_artifacts: int = 2) -> np.ndarray:
    """
    使用FastICA进行伪迹去除
    假设前n_artifacts个成分包含最多伪迹（基于方差）
    """
    n_trials, n_channels, n_times = data.shape
    cleaned_data = np.zeros_like(data)
    
    for i in range(n_trials):
        trial_data = data[i]  # (n_channels, n_times)
        
        # 应用ICA
        n_comp = min(n_components, n_channels)
        ica = FastICA(n_components=n_comp, random_state=42, max_iter=1000, tol=0.01)
        
        try:
            # 转置为 (n_times, n_channels) 进行ICA
            sources = ica.fit_transform(trial_data.T)  # (n_times, n_components)
            mixing = ica.mixing_  # (n_channels, n_components)
            
            # 计算每个成分的方差（假设高方差成分更可能是伪迹）
            variances = np.var(sources, axis=0)
            artifact_indices = np.argsort(variances)[-n_artifacts:]  # 选择方差最大的成分
            
            # 将伪迹成分置零
            sources_cleaned = sources.copy()
            sources_cleaned[:, artifact_indices] = 0
            
            # 重建信号
            reconstructed = np.dot(sources_cleaned, mixing.T)  # (n_times, n_channels)
            cleaned_data[i] = reconstructed.T
            
        except Exception as e:
            # 如果ICA失败，保留原始数据
            cleaned_data[i] = trial_data
    
    return cleaned_data


def process_subject_data(subject_id: int, use_ica: bool = True, 
                         use_bandpass: bool = True) -> dict:
    """处理单个受试者的数据"""
    results = {}
    
    for mode in ['train', 'test']:
        X, y = load_mat_data(subject_id, mode)
        print(f"  {mode}: 原始数据形状 {X.shape}")
        
        # 带通滤波
        if use_bandpass:
            X = bandpass_filter(X)
            print(f"  {mode}: 带通滤波 8-30Hz 完成")
        
        # ICA去伪迹
        if use_ica:
            X = apply_ica_cleaning(X, n_components=20, n_artifacts=2)
            print(f"  {mode}: ICA去伪迹完成")
        
        results[mode] = {'data': X, 'label': y}
    
    return results


def save_mat_data(results: dict, output_dir: Path, subject_id: int):
    """保存处理后的数据为mat格式"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for mode, data in results.items():
        suffix = 'T' if mode == 'train' else 'E'
        output_file = output_dir / f"A{subject_id:02d}{suffix}.mat"
        
        # 添加通道维度（CTNet需要）
        X = data['data']
        if X.ndim == 3:
            X = np.expand_dims(X, axis=1)  # (trials, 1, channels, times)
            X = X.squeeze(1)  # 保持 (trials, channels, times)
        
        scipy.io.savemat(str(output_file), {
            'data': X,
            'label': data['label'].reshape(-1, 1)
        })
        print(f"  保存: {output_file}")


def run_ctnet_training(data_dir: Path, result_name: str, epochs: int = 300):
    """运行CTNet训练"""
    print(f"\n{'='*70}")
    print(f"开始CTNet训练: {result_name}")
    print(f"数据目录: {data_dir}")
    print(f"{'='*70}")
    
    sys.path.insert(0, str(SCRIPT_DIR))
    
    from CTNet_model import ExP, calMetrics, numberClassChannel, get_class_labels, plot_confusion_matrix
    import torch
    import random
    import pandas as pd
    from pandas import ExcelWriter
    
    # 训练参数
    N_SUBJECT = 9
    N_AUG = 3
    N_SEG = 8
    data_dir_str = str(data_dir) + "/"
    
    # 创建结果目录
    if not os.path.exists(result_name):
        os.makedirs(result_name)
    
    result_write_metric = ExcelWriter(result_name + "/result_metric.xlsx")
    process_write = ExcelWriter(result_name + "/process_train.xlsx")
    pred_true_write = ExcelWriter(result_name + "/pred_true.xlsx")
    
    subjects_result = []
    best_epochs = []
    yt, yp = None, None
    
    for i in range(N_SUBJECT):
        starttime = datetime.now()
        seed_n = np.random.randint(2024)
        print(f'Subject {i+1}: Seed = {seed_n}')
        random.seed(seed_n)
        np.random.seed(seed_n)
        torch.manual_seed(seed_n)
        
        exp = ExP(i + 1, data_dir_str, result_name, epochs, N_AUG, N_SEG, None, 
                  evaluate_mode='subject-dependent',
                  heads=2, 
                  emb_size=40,
                  depth=6, 
                  dataset_type='A',
                  validate_ratio=0.2,
                  early_stopping=False,  # 取消早停
                  patience=50,
                  min_delta=0.0001,
                  verbose=True,
                  plot_training=True,  # 显示训练曲线
                  batch_size=72,
                  learning_rate=0.001,
                  )
        testAcc, Y_true, Y_pred, df_process, best_epoch = exp.train()
        true_cpu = Y_true.cpu().numpy().astype(int)
        pred_cpu = Y_pred.cpu().numpy().astype(int)
        
        # 获取类别标签
        number_class, _ = numberClassChannel('A')
        class_labels = get_class_labels('A', number_class)
        
        # 保存预测结果
        true_labels = [class_labels[t] for t in true_cpu]
        pred_labels = [class_labels[p] for p in pred_cpu]
        df_pred_true = pd.DataFrame({
            'true_class': true_cpu,
            'pred_class': pred_cpu,
            'true_label': true_labels,
            'pred_label': pred_labels,
            'correct': (true_cpu == pred_cpu).astype(int)
        })
        df_pred_true.to_excel(pred_true_write, sheet_name=str(i+1))
        
        # 生成混淆矩阵
        plot_confusion_matrix(true_cpu, pred_cpu, class_labels, 
                             result_name, i+1, 'A', verbose=True)
        
        accuracy, precision, recall, f1, kappa = calMetrics(true_cpu, pred_cpu)
        subject_result = {'accuray': accuracy*100,
                          'precision': precision*100,
                          'recall': recall*100,
                          'f1': f1*100, 
                          'kappa': kappa*100
                          }
        subjects_result.append(subject_result)
        df_process.to_excel(process_write, sheet_name=str(i+1))
        best_epochs.append(best_epoch)
        
        print(f'Subject {i+1}: 最佳准确率 {testAcc:.4f} | Kappa: {kappa:.4f}')
        endtime = datetime.now()
        print(f'Subject {i+1}: 训练耗时 {endtime - starttime}')
        
        if yt is None:
            yt = Y_true
            yp = Y_pred
        else:
            yt = torch.cat((yt, Y_true))
            yp = torch.cat((yp, Y_pred))
    
    df_result = pd.DataFrame(subjects_result)
    
    # 生成总体混淆矩阵
    if yt is not None:
        yt_all = yt.cpu().numpy().astype(int)
        yp_all = yp.cpu().numpy().astype(int)
        plot_confusion_matrix(yt_all, yp_all, class_labels, 
                             result_name, 'all_subjects', 'A', 
                             title_suffix=' (All Subjects)', verbose=True)
    
    process_write.close()
    pred_true_write.close()
    
    print(f'\n最终结果: 平均准确率 {df_result["accuray"].mean():.2f}% | 平均 Kappa {df_result["kappa"].mean():.4f}')
    print(f"最佳 Epochs: {best_epochs}")
    
    df_result.to_excel(result_write_metric, sheet_name='metrics')
    result_write_metric.close()
    
    return {'result_metric': df_result, 'best_epochs': best_epochs}


def compare_results(result_with_ica: dict, result_without_ica: dict):
    """对比两组结果"""
    print("\n" + "="*70)
    print("ICA效果对比结果")
    print("="*70)
    
    if result_with_ica is None or result_without_ica is None:
        print("错误: 缺少训练结果，无法对比")
        return None
    
    df_with_ica = result_with_ica.get('result_metric')
    df_without_ica = result_without_ica.get('result_metric')
    
    if df_with_ica is None or df_without_ica is None:
        print("错误: 无法提取指标数据")
        return None
    
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
    
    print(f"\n统计分析:")
    print(f"  - 使用ICA后准确率提升的受试者: {sum(1 for d in improvements if d > 0)}/9")
    print(f"  - 使用ICA后准确率下降的受试者: {sum(1 for d in improvements if d < 0)}/9")
    print(f"  - 平均变化: {np.mean(improvements):.2f}%")
    print(f"  - 最大提升: {max(improvements):.2f}%")
    print(f"  - 最大下降: {min(improvements):.2f}%")
    
    comparison = {
        "timestamp": datetime.now().isoformat(),
        "avg_accuracy_without_ica": float(avg_without),
        "avg_accuracy_with_ica": float(avg_with),
        "improvement": float(avg_diff),
        "per_subject_improvement": [float(d) for d in improvements],
    }
    
    result_file = OUTPUT_DIR / "comparison_result.json"
    with open(result_file, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n对比结果已保存: {result_file}")
    
    return comparison


def main(training_epochs: int = 300, skip_preprocessing: bool = False):
    """
    主函数
    
    Parameters:
    -----------
    training_epochs : int
        训练轮数
    skip_preprocessing : bool
        是否跳过预处理（使用已有数据）
    """
    print("="*70)
    print("ICA 去伪迹效果对比实验")
    print("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data_dir_no_ica = OUTPUT_DIR / "data_no_ica"
    data_dir_with_ica = OUTPUT_DIR / "data_with_ica"
    
    if not skip_preprocessing:
        # 处理数据（不使用ICA，只带通滤波）
        print("\n" + "="*70)
        print("步骤1: 预处理数据（带通滤波，不使用ICA）")
        print("="*70)
        
        for subject_id in range(1, 10):
            print(f"\n处理受试者 {subject_id}...")
            results = process_subject_data(subject_id, use_ica=False, use_bandpass=True)
            save_mat_data(results, data_dir_no_ica, subject_id)
        
        # 处理数据（使用ICA）
        print("\n" + "="*70)
        print("步骤2: 预处理数据（带通滤波 + ICA）")
        print("="*70)
        
        for subject_id in range(1, 10):
            print(f"\n处理受试者 {subject_id}...")
            results = process_subject_data(subject_id, use_ica=True, use_bandpass=True)
            save_mat_data(results, data_dir_with_ica, subject_id)
    else:
        print("\n跳过预处理，使用已有数据...")
    
    # 训练CTNet
    print("\n" + "="*70)
    print("步骤3: 训练CTNet（不使用ICA数据）")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_no_ica = run_ctnet_training(
        data_dir_no_ica, 
        f"CTNet_no_ica_{timestamp}",
        epochs=training_epochs
    )
    
    print("\n" + "="*70)
    print("步骤4: 训练CTNet（使用ICA数据）")
    print("="*70)
    
    result_with_ica = run_ctnet_training(
        data_dir_with_ica,
        f"CTNet_with_ica_{timestamp}",
        epochs=training_epochs
    )
    
    # 对比结果
    compare_results(result_with_ica, result_no_ica)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICA效果对比实验')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮数')
    parser.add_argument('--skip_preprocess', action='store_true', help='跳过预处理')
    
    args = parser.parse_args()
    
    main(training_epochs=args.epochs, skip_preprocessing=args.skip_preprocess)

