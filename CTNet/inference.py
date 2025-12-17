#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTNet 单次推理脚本
用于加载已训练模型，对测试数据进行推理，并生成混淆矩阵可视化
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('TkAgg')  # 使用交互式后端
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    sns = None

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score
import scipy.io

# 导入模型类定义（加载整个模型时需要）
from CTNet_model import (
    EEGTransformer, BranchEEGNetTransformer, PatchEmbeddingCNN,
    MultiHeadAttention, FeedForwardBlock, ClassificationHead,
    ResidualAdd, TransformerEncoderBlock, TransformerEncoder,
    PositioinalEncoding
)

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Inference] Using device: {device}")

# 类别标签映射
CLASS_LABELS = {
    'A': {
        0: 'Left Hand\n(左手)',
        1: 'Right Hand\n(右手)',
        2: 'Both Feet\n(双脚)',
        3: 'Tongue\n(舌头)'
    },
    'B': {
        0: 'Left Hand\n(左手)',
        1: 'Right Hand\n(右手)'
    }
}

# 简短标签（用于输出）
CLASS_LABELS_SHORT = {
    'A': {0: 'Left', 1: 'Right', 2: 'Feet', 3: 'Tongue'},
    'B': {0: 'Left', 1: 'Right'}
}


def load_model(model_path, device=device):
    """
    加载已训练的模型
    
    Parameters:
    -----------
    model_path : str
        模型文件路径 (.pth)
    device : torch.device
        设备
    
    Returns:
    --------
    model : nn.Module
        加载的模型
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    print(f"✓ 模型已加载: {model_path}")
    return model


def load_test_data(data_dir, dataset_type, subject_id, normalize_params=None):
    """
    加载测试数据
    
    Parameters:
    -----------
    data_dir : str
        数据目录
    dataset_type : str
        数据集类型 ('A' 或 'B')
    subject_id : int
        受试者ID (1-9)
    normalize_params : tuple, optional
        归一化参数 (mean, std)，如果为None则使用测试数据自身的统计量
    
    Returns:
    --------
    data : torch.Tensor
        测试数据
    labels : torch.Tensor
        测试标签
    """
    # 加载测试文件
    test_file = os.path.join(data_dir, f'{dataset_type}{subject_id:02d}E.mat')
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"测试文件不存在: {test_file}")
    
    data_mat = scipy.io.loadmat(test_file)
    data = data_mat['data']
    labels = data_mat['label']
    
    # 添加通道维度
    data = np.expand_dims(data, axis=1)
    labels = labels.flatten()
    
    # 归一化
    if normalize_params is not None:
        mean, std = normalize_params
    else:
        # 可选：加载训练数据计算归一化参数
        train_file = os.path.join(data_dir, f'{dataset_type}{subject_id:02d}T.mat')
        if os.path.exists(train_file):
            train_mat = scipy.io.loadmat(train_file)
            train_data = np.expand_dims(train_mat['data'], axis=1)
            mean = np.mean(train_data)
            std = np.std(train_data)
        else:
            mean = np.mean(data)
            std = np.std(data)
    
    data = (data - mean) / std
    
    # 转换为Tensor
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels - 1).long()  # 标签从0开始
    
    print(f"✓ 数据已加载: {test_file}")
    print(f"  数据形状: {data.shape}, 标签数量: {len(labels)}")
    
    return data, labels


def inference(model, data, batch_size=64):
    """
    执行推理
    
    Parameters:
    -----------
    model : nn.Module
        模型
    data : torch.Tensor
        输入数据
    batch_size : int
        批次大小
    
    Returns:
    --------
    predictions : numpy.ndarray
        预测结果
    probabilities : numpy.ndarray
        各类别的概率
    """
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size].to(device)
            _, outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


def plot_confusion_matrix_interactive(y_true, y_pred, class_labels, dataset_type, 
                                      subject_id=None, save_path=None, show=True):
    """
    绘制并显示混淆矩阵（交互式）
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        真实标签
    y_pred : numpy.ndarray
        预测标签
    class_labels : dict
        类别标签映射
    dataset_type : str
        数据集类型
    subject_id : int, optional
        受试者ID
    save_path : str, optional
        保存路径
    show : bool
        是否显示图形
    """
    num_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # 计算指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 归一化混淆矩阵
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_normalized = np.nan_to_num(cm_normalized)
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
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
    
    # 在矩阵上叠加数值
    for i in range(num_classes):
        for j in range(num_classes):
            percentage = cm_normalized[i, j]
            count = cm[i, j]
            text_color = 'white' if percentage > 50 else 'black'
            ax.text(j+0.5, i+0.7, f'{percentage:.1f}%', 
                   ha='center', va='center', 
                   fontsize=12, color=text_color, weight='bold')
            ax.text(j+0.5, i+0.3, f'({count})', 
                   ha='center', va='center', 
                   fontsize=10, color=text_color)
    
    # 标题和标签
    title = f'Confusion Matrix'
    if subject_id is not None:
        title += f' - Subject {subject_id}'
    title += f'\nAccuracy: {accuracy:.2%} ({accuracy*100:.2f}%)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {save_path}")
    
    # 显示
    if show:
        plt.show()
    else:
        plt.close()
    
    return cm, accuracy


def print_detailed_results(y_true, y_pred, class_labels_short):
    """
    打印详细的预测结果
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        真实标签
    y_pred : numpy.ndarray
        预测标签
    class_labels_short : dict
        简短类别标签
    """
    num_classes = len(class_labels_short)
    
    # 计算各种指标
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print("\n" + "="*70)
    print("推理结果汇总")
    print("="*70)
    print(f"总样本数: {len(y_true)}")
    print(f"正确预测: {np.sum(y_true == y_pred)}")
    print(f"错误预测: {np.sum(y_true != y_pred)}")
    print("-"*70)
    print(f"准确率 (Accuracy):  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"精确率 (Precision): {precision:.4f} ({precision*100:.2f}%)")
    print(f"召回率 (Recall):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1分数 (F1-Score):  {f1:.4f} ({f1*100:.2f}%)")
    print(f"Kappa系数:          {kappa:.4f}")
    print("-"*70)
    
    # 每个类别的统计
    print("\n各类别统计:")
    print("-"*50)
    print(f"{'类别':<15} {'样本数':<8} {'正确':<8} {'准确率':<10}")
    print("-"*50)
    
    for cls_id, cls_name in class_labels_short.items():
        mask = (y_true == cls_id)
        total = np.sum(mask)
        correct = np.sum((y_true == cls_id) & (y_pred == cls_id))
        cls_acc = correct / total if total > 0 else 0.0
        print(f"{cls_name:<15} {total:<8} {correct:<8} {cls_acc:.2%}")
    
    print("="*70 + "\n")


def run_inference(model_path, data_dir, dataset_type, subject_id, 
                  save_dir=None, show_plot=True):
    """
    运行单次推理的完整流程
    
    Parameters:
    -----------
    model_path : str
        模型文件路径
    data_dir : str
        数据目录
    dataset_type : str
        数据集类型 ('A' 或 'B')
    subject_id : int
        受试者ID
    save_dir : str, optional
        结果保存目录
    show_plot : bool
        是否显示图形
    
    Returns:
    --------
    results : dict
        推理结果字典
    """
    print("\n" + "="*70)
    print(f"CTNet 单次推理")
    print(f"数据集: {dataset_type}, 受试者: {subject_id}")
    print("="*70 + "\n")
    
    # 1. 加载模型
    model = load_model(model_path)
    
    # 2. 加载数据
    data, labels = load_test_data(data_dir, dataset_type, subject_id)
    
    # 3. 执行推理
    print("\n执行推理...")
    predictions, probabilities = inference(model, data)
    print(f"✓ 推理完成，共 {len(predictions)} 个样本")
    
    # 4. 获取类别标签
    class_labels = CLASS_LABELS.get(dataset_type, CLASS_LABELS['A'])
    class_labels_short = CLASS_LABELS_SHORT.get(dataset_type, CLASS_LABELS_SHORT['A'])
    
    # 5. 打印详细结果
    y_true = labels.numpy()
    y_pred = predictions
    print_detailed_results(y_true, y_pred, class_labels_short)
    
    # 6. 绘制混淆矩阵
    save_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'confusion_matrix_inference_subject_{subject_id}.png')
    
    cm, accuracy = plot_confusion_matrix_interactive(
        y_true, y_pred, class_labels, dataset_type, 
        subject_id=subject_id, save_path=save_path, show=show_plot
    )
    
    # 7. 保存预测结果
    if save_dir:
        results_df = pd.DataFrame({
            'sample_id': range(len(y_true)),
            'true_class': y_true,
            'pred_class': y_pred,
            'true_label': [class_labels_short[t] for t in y_true],
            'pred_label': [class_labels_short[p] for p in y_pred],
            'correct': (y_true == y_pred).astype(int),
            'confidence': probabilities.max(axis=1)
        })
        
        results_file = os.path.join(save_dir, f'inference_results_subject_{subject_id}.xlsx')
        results_df.to_excel(results_file, index=False)
        print(f"✓ 预测结果已保存: {results_file}")
    
    # 8. 返回结果
    results = {
        'y_true': y_true,
        'y_pred': y_pred,
        'probabilities': probabilities,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='CTNet 单次推理')
    parser.add_argument('--model', '-m', type=str, required=True, help='模型文件路径 (.pth)')
    parser.add_argument('--data_dir', '-d', type=str, default='./mymat_raw/', help='数据目录')
    parser.add_argument('--dataset', '-t', type=str, default='A', choices=['A', 'B'], help='数据集类型')
    parser.add_argument('--subject', '-s', type=int, default=1, help='受试者ID (1-9)')
    parser.add_argument('--save_dir', '-o', type=str, default='./inference_results/', help='结果保存目录')
    parser.add_argument('--no_show', action='store_true', help='不显示图形（仅保存）')
    
    args = parser.parse_args()
    
    results = run_inference(
        model_path=args.model,
        data_dir=args.data_dir,
        dataset_type=args.dataset,
        subject_id=args.subject,
        save_dir=args.save_dir,
        show_plot=not args.no_show
    )
    
    return results


if __name__ == "__main__":
    # 如果直接运行脚本，使用命令行参数
    if len(sys.argv) > 1:
        main()
    else:
        # 交互式示例
        print("="*70)
        print("CTNet 单次推理示例")
        print("="*70)
        print("\n使用方法:")
        print("  python inference.py -m <模型路径> -d <数据目录> -t <数据集A/B> -s <受试者ID>")
        print("\n示例:")
        print("  python inference.py -m ./A_heads_2_depth_6/model_1.pth -d ./mymat_raw/ -t A -s 1")
        print("\n或者在Python中导入使用:")
        print("  from inference import run_inference")
        print("  results = run_inference(model_path, data_dir, dataset_type, subject_id)")
        print("="*70)
        
        # 可选：运行示例
        example_model = './A_heads_2_depth_6/model_1.pth'
        if os.path.exists(example_model):
            print(f"\n检测到示例模型: {example_model}")
            response = input("是否运行示例推理? (y/n): ")
            if response.lower() == 'y':
                results = run_inference(
                    model_path=example_model,
                    data_dir='./mymat_raw/',
                    dataset_type='A',
                    subject_id=1,
                    save_dir='./inference_results/',
                    show_plot=True
                )

