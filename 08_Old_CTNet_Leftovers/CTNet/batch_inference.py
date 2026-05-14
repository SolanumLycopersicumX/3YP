#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTNet 批量推理脚本
对指定目录下的所有模型进行推理并生成混淆矩阵
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import scipy.io
from datetime import datetime

# 导入模型类定义
from CTNet_model import (
    EEGTransformer, BranchEEGNetTransformer, PatchEmbeddingCNN,
    MultiHeadAttention, FeedForwardBlock, ClassificationHead,
    ResidualAdd, TransformerEncoderBlock, TransformerEncoder,
    PositioinalEncoding
)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Batch Inference] Using device: {device}")

# 类别标签
CLASS_LABELS = {
    'A': {0: 'Left Hand', 1: 'Right Hand', 2: 'Both Feet', 3: 'Tongue'},
    'B': {0: 'Left Hand', 1: 'Right Hand'}
}


def load_model(model_path):
    """加载模型"""
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


def load_test_data(data_dir, dataset_type, subject_id):
    """加载测试数据"""
    test_file = os.path.join(data_dir, f'{dataset_type}{subject_id:02d}E.mat')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    data_mat = scipy.io.loadmat(test_file)
    data = np.expand_dims(data_mat['data'], axis=1)
    labels = data_mat['label'].flatten()
    
    # 加载训练数据计算归一化参数
    train_file = os.path.join(data_dir, f'{dataset_type}{subject_id:02d}T.mat')
    if os.path.exists(train_file):
        train_mat = scipy.io.loadmat(train_file)
        train_data = np.expand_dims(train_mat['data'], axis=1)
        mean, std = np.mean(train_data), np.std(train_data)
    else:
        mean, std = np.mean(data), np.std(data)
    
    data = (data - mean) / std
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels - 1).long()
    
    return data, labels


def inference(model, data, batch_size=64):
    """执行推理"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            _, outputs = model(batch)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
    
    return np.array(predictions)


def plot_confusion_matrix_single(y_true, y_pred, class_labels, subject_id, accuracy, ax):
    """绘制单个混淆矩阵"""
    num_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_normalized = np.nan_to_num(cm_normalized)
    
    if SEABORN_AVAILABLE:
        sns.heatmap(cm_normalized, annot=False, fmt='.1f', cmap='Blues', 
                    cbar=False, ax=ax,
                    xticklabels=[class_labels[i] for i in range(num_classes)],
                    yticklabels=[class_labels[i] for i in range(num_classes)],
                    linewidths=0.5, linecolor='gray')
    else:
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([class_labels[i] for i in range(num_classes)])
        ax.set_yticklabels([class_labels[i] for i in range(num_classes)])
    
    # 添加数值标注
    for i in range(num_classes):
        for j in range(num_classes):
            percentage = cm_normalized[i, j]
            count = cm[i, j]
            text_color = 'white' if percentage > 50 else 'black'
            ax.text(j+0.5, i+0.65, f'{percentage:.0f}%', 
                   ha='center', va='center', fontsize=8, color=text_color, weight='bold')
            ax.text(j+0.5, i+0.35, f'({count})', 
                   ha='center', va='center', fontsize=6, color=text_color)
    
    ax.set_title(f'Subject {subject_id}\nAcc: {accuracy:.1f}%', fontsize=10, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    return cm


def plot_all_confusion_matrices(all_results, class_labels, save_path, dataset_type):
    """绘制所有受试者的混淆矩阵（3x3网格）"""
    num_subjects = len(all_results)
    
    # 计算网格大小
    cols = 3
    rows = (num_subjects + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4*rows))
    axes = axes.flatten() if num_subjects > 1 else [axes]
    
    for idx, (subject_id, result) in enumerate(all_results.items()):
        plot_confusion_matrix_single(
            result['y_true'], result['y_pred'], 
            class_labels, subject_id, 
            result['accuracy'] * 100, axes[idx]
        )
    
    # 隐藏多余的子图
    for idx in range(num_subjects, len(axes)):
        axes[idx].axis('off')
    
    # 计算平均准确率
    avg_acc = np.mean([r['accuracy'] for r in all_results.values()]) * 100
    
    fig.suptitle(f'Confusion Matrices - All Subjects (Dataset {dataset_type})\n'
                 f'Average Accuracy: {avg_acc:.2f}%', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 所有混淆矩阵已保存: {save_path}")
    
    return fig


def plot_aggregated_confusion_matrix(all_results, class_labels, save_path):
    """绘制汇总混淆矩阵（所有受试者数据合并）"""
    # 合并所有预测结果
    all_y_true = np.concatenate([r['y_true'] for r in all_results.values()])
    all_y_pred = np.concatenate([r['y_pred'] for r in all_results.values()])
    
    num_classes = len(class_labels)
    cm = confusion_matrix(all_y_true, all_y_pred, labels=list(range(num_classes)))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_normalized = np.nan_to_num(cm_normalized)
    
    accuracy = accuracy_score(all_y_true, all_y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if SEABORN_AVAILABLE:
        sns.heatmap(cm_normalized, annot=False, fmt='.1f', cmap='Blues', 
                    cbar_kws={'label': 'Percentage (%)'},
                    xticklabels=[class_labels[i] for i in range(num_classes)],
                    yticklabels=[class_labels[i] for i in range(num_classes)],
                    linewidths=0.5, linecolor='gray', ax=ax)
    else:
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.colorbar(im, ax=ax, label='Percentage (%)')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        ax.set_xticklabels([class_labels[i] for i in range(num_classes)])
        ax.set_yticklabels([class_labels[i] for i in range(num_classes)])
    
    # 添加数值标注
    for i in range(num_classes):
        for j in range(num_classes):
            percentage = cm_normalized[i, j]
            count = cm[i, j]
            text_color = 'white' if percentage > 50 else 'black'
            ax.text(j+0.5, i+0.7, f'{percentage:.1f}%', 
                   ha='center', va='center', fontsize=12, color=text_color, weight='bold')
            ax.text(j+0.5, i+0.3, f'({count})', 
                   ha='center', va='center', fontsize=10, color=text_color)
    
    total_samples = len(all_y_true)
    ax.set_title(f'Aggregated Confusion Matrix (All {len(all_results)} Subjects)\n'
                 f'Total Samples: {total_samples} | Accuracy: {accuracy:.2%}', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 汇总混淆矩阵已保存: {save_path}")
    
    return fig, cm, accuracy


def batch_inference(model_dir, data_dir, dataset_type='A', save_dir=None, show_plot=True):
    """
    批量推理所有模型
    
    Parameters:
    -----------
    model_dir : str
        模型目录路径
    data_dir : str
        数据目录路径
    dataset_type : str
        数据集类型 ('A' 或 'B')
    save_dir : str
        结果保存目录
    show_plot : bool
        是否显示图形
    """
    print("\n" + "="*70)
    print(f"CTNet 批量推理")
    print(f"模型目录: {model_dir}")
    print(f"数据目录: {data_dir}")
    print(f"数据集类型: {dataset_type}")
    print("="*70 + "\n")
    
    # 查找所有模型文件
    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pth')])
    
    if not model_files:
        print(f"错误: 在 {model_dir} 中未找到模型文件 (.pth)")
        return None
    
    print(f"找到 {len(model_files)} 个模型文件:")
    for f in model_files:
        print(f"  - {f}")
    print()
    
    # 获取类别标签
    class_labels = CLASS_LABELS.get(dataset_type, CLASS_LABELS['A'])
    
    # 存储所有结果
    all_results = {}
    summary_data = []
    
    # 逐个推理
    for model_file in model_files:
        # 从文件名提取受试者ID
        # 假设文件名格式为 model_X.pth 或 subject_X_model.pth
        try:
            subject_id = int(''.join(filter(str.isdigit, model_file.split('.')[0])))
        except:
            subject_id = model_files.index(model_file) + 1
        
        model_path = os.path.join(model_dir, model_file)
        
        print(f"[{model_file}] 推理受试者 {subject_id}...")
        
        try:
            # 加载模型
            model = load_model(model_path)
            
            # 加载数据
            data, labels = load_test_data(data_dir, dataset_type, subject_id)
            
            # 执行推理
            predictions = inference(model, data)
            y_true = labels.numpy()
            y_pred = predictions
            
            # 计算指标
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro')
            recall = recall_score(y_true, y_pred, average='macro')
            f1 = f1_score(y_true, y_pred, average='macro')
            kappa = cohen_kappa_score(y_true, y_pred)
            
            # 保存结果
            all_results[subject_id] = {
                'y_true': y_true,
                'y_pred': y_pred,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'kappa': kappa
            }
            
            summary_data.append({
                'Subject': subject_id,
                'Accuracy': accuracy * 100,
                'Precision': precision * 100,
                'Recall': recall * 100,
                'F1-Score': f1 * 100,
                'Kappa': kappa * 100
            })
            
            print(f"  ✓ 准确率: {accuracy:.2%}")
            
        except Exception as e:
            print(f"  ✗ 错误: {str(e)}")
            continue
    
    if not all_results:
        print("错误: 所有推理都失败了")
        return None
    
    # 创建保存目录
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f'./batch_inference_results_{timestamp}/'
    os.makedirs(save_dir, exist_ok=True)
    
    # 打印汇总
    print("\n" + "="*70)
    print("推理结果汇总")
    print("="*70)
    
    summary_df = pd.DataFrame(summary_data)
    
    # 添加平均值行
    mean_row = summary_df.mean(numeric_only=True).to_dict()
    mean_row['Subject'] = 'Mean'
    std_row = summary_df.std(numeric_only=True).to_dict()
    std_row['Subject'] = 'Std'
    
    summary_df = pd.concat([summary_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)
    
    print(summary_df.to_string(index=False))
    print("="*70)
    
    # 保存汇总表格
    summary_file = os.path.join(save_dir, 'inference_summary.xlsx')
    summary_df.to_excel(summary_file, index=False)
    print(f"\n✓ 汇总表格已保存: {summary_file}")
    
    # 绘制所有混淆矩阵
    all_cm_path = os.path.join(save_dir, 'confusion_matrices_all_subjects.png')
    fig1 = plot_all_confusion_matrices(all_results, class_labels, all_cm_path, dataset_type)
    
    # 绘制汇总混淆矩阵
    agg_cm_path = os.path.join(save_dir, 'confusion_matrix_aggregated.png')
    fig2, agg_cm, agg_acc = plot_aggregated_confusion_matrix(all_results, class_labels, agg_cm_path)
    
    print(f"\n汇总准确率: {agg_acc:.2%}")
    
    if show_plot:
        plt.show()
    else:
        plt.close('all')
    
    return {
        'all_results': all_results,
        'summary': summary_df,
        'aggregated_accuracy': agg_acc,
        'aggregated_confusion_matrix': agg_cm
    }


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CTNet 批量推理')
    parser.add_argument('--model_dir', '-m', type=str, required=True, help='模型目录路径')
    parser.add_argument('--data_dir', '-d', type=str, default='./mymat_raw/', help='数据目录')
    parser.add_argument('--dataset', '-t', type=str, default='A', choices=['A', 'B'], help='数据集类型')
    parser.add_argument('--save_dir', '-o', type=str, default=None, help='结果保存目录')
    parser.add_argument('--no_show', action='store_true', help='不显示图形')
    
    args = parser.parse_args()
    
    results = batch_inference(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        save_dir=args.save_dir,
        show_plot=not args.no_show
    )
    
    return results


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # 默认运行示例
        print("使用方法:")
        print("  python batch_inference.py -m <模型目录> -d <数据目录> -t <数据集A/B>")
        print("\n示例:")
        print("  python batch_inference.py -m ./models/new -d ./mymat_raw/ -t A")




