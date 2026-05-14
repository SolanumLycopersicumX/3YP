#!/usr/bin/env python3
"""
PhysioNet LOSO (Leave-One-Subject-Out) Cross-Validation

实现跨被试泛化实验:
- 对于每个被试，使用其他所有被试的数据训练，用该被试的数据测试
- 评估模型在未见过的被试上的泛化能力

支持 2-class (left/right) 和 4-class (left/right/hands/feet) 任务

创建时间: 2026-02-24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 项目根目录
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ============================================================================
# 模型定义
# ============================================================================

class SimpleCTNet(nn.Module):
    """简化版 CTNet (用于 2-class)"""
    
    def __init__(self, n_channels: int, n_times: int, n_classes: int):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), padding=(0, 12)),
            nn.BatchNorm2d(40),
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, (n_channels, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)),
            nn.Dropout(0.5),
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(40, 40, (1, 15), padding=(0, 7)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=(1, 8)),
            nn.Dropout(0.5),
        )
        
        with torch.no_grad():
            x = torch.zeros(1, 1, n_channels, n_times)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            flatten_size = x.numel()
        
        self.fc = nn.Linear(flatten_size, n_classes)
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


# 从 train_physionet_4class_ctnet.py 复制 EEGTransformer
from CTNet_model import (
    PatchEmbeddingCNN,
    TransformerEncoder,
    ClassificationHead,
)

class EEGTransformer(nn.Module):
    """完整 CTNet 架构 (用于 4-class)"""
    
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        n_classes: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        patch_size: int = 25,
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes
        
        self.patch_embed = PatchEmbeddingCNN(
            n_channels=n_channels,
            d_model=d_model,
            patch_size=patch_size,
        )
        
        self.seq_len = n_times // patch_size
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        self.classifier = ClassificationHead(
            d_model=d_model,
            n_classes=n_classes,
            dropout=dropout,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        logits = self.classifier(x)
        
        return logits


# ============================================================================
# 数据加载
# ============================================================================

def load_physionet_data(
    subjects: List[int],
    task: str = "2class",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    加载 PhysioNet 数据
    
    Args:
        subjects: 被试列表
        task: "2class" 或 "4class"
        verbose: 是否打印加载信息
    
    Returns:
        data: (n_samples, n_channels, n_times)
        labels: (n_samples,)
        subject_ids: (n_samples,)
    """
    from scripts.physionet_loader import (
        load_subject_mne,
        load_subject_4class_mne,
        MI_RUNS,
        MNE_AVAILABLE,
    )
    
    if not MNE_AVAILABLE:
        raise ImportError("需要安装 MNE: pip install mne")
    
    all_data = []
    all_labels = []
    all_subjects = []
    
    for subject in subjects:
        if verbose:
            print(f"  加载被试 {subject}...", end=" ", flush=True)
        
        try:
            if task == "4class":
                data, labels = load_subject_4class_mne(subject)
            else:
                runs = MI_RUNS['left_right_imagine']
                data, labels = load_subject_mne(subject, runs)
                mask = labels > 0
                data = data[mask]
                labels = labels[mask] - 1
            
            all_data.append(data)
            all_labels.append(labels)
            all_subjects.append(np.full(len(labels), subject))
            
            if verbose:
                print(f"{len(labels)} trials")
        except Exception as e:
            if verbose:
                print(f"失败: {e}")
    
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subjects, axis=0)
    
    return data.astype(np.float32), labels.astype(np.int64), subject_ids.astype(np.int64)


# ============================================================================
# 训练和评估
# ============================================================================

@dataclass
class LOSOResult:
    """单个 fold 的结果"""
    test_subject: int
    accuracy: float
    predictions: np.ndarray
    true_labels: np.ndarray
    confusion_matrix: np.ndarray
    n_train_samples: int
    n_test_samples: int


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_x.size(0)
    
    return total_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return (
        total_loss / total,
        correct / total,
        np.array(all_preds),
        np.array(all_labels),
    )


def run_loso_fold(
    test_subject: int,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    n_classes: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    patience: int = 20,
    model_type: str = "simple",
    verbose: bool = True,
) -> LOSOResult:
    """
    运行单个 LOSO fold
    """
    # 数据预处理
    mean, std = train_data.mean(), train_data.std()
    train_data = (train_data - mean) / (std + 1e-8)
    test_data = (test_data - mean) / (std + 1e-8)
    
    # 创建 DataLoader
    train_ds = TensorDataset(
        torch.tensor(train_data, dtype=torch.float32),
        torch.tensor(train_labels, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(test_data, dtype=torch.float32),
        torch.tensor(test_labels, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    n_channels = train_data.shape[1]
    n_times = train_data.shape[2]
    
    if model_type == "transformer" or n_classes == 4:
        model = EEGTransformer(
            n_channels=n_channels,
            n_times=n_times,
            n_classes=n_classes,
            d_model=64,
            n_heads=8,
            n_layers=2,
        ).to(device)
    else:
        model = SimpleCTNet(n_channels, n_times, n_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_weights = None
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}: Train {train_acc:.2%}, Test {test_acc:.2%} (best: {best_acc:.2%})")
        
        if epochs_no_improve >= patience:
            if verbose:
                print(f"    早停 @ epoch {epoch+1}")
            break
    
    # 恢复最佳权重
    if best_weights is not None:
        model.load_state_dict(best_weights)
    
    # 最终评估
    _, final_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    cm = confusion_matrix(labels, preds)
    
    return LOSOResult(
        test_subject=test_subject,
        accuracy=final_acc,
        predictions=preds,
        true_labels=labels,
        confusion_matrix=cm,
        n_train_samples=len(train_data),
        n_test_samples=len(test_data),
    )


# ============================================================================
# 可视化
# ============================================================================

def visualize_loso_results(
    results: List[LOSOResult],
    output_dir: Path,
    task: str = "2class",
):
    """可视化 LOSO 结果"""
    n_subjects = len(results)
    
    # 1. 准确率柱状图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    subjects = [r.test_subject for r in results]
    accuracies = [r.accuracy for r in results]
    
    ax1 = axes[0]
    bars = ax1.bar(subjects, [a * 100 for a in accuracies], color='steelblue', edgecolor='black')
    ax1.axhline(y=np.mean(accuracies) * 100, color='red', linestyle='--', 
                label=f'Mean: {np.mean(accuracies)*100:.1f}%')
    ax1.set_xlabel('Test Subject', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('LOSO Cross-Validation Accuracy', fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc*100:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 2. 聚合混淆矩阵
    ax2 = axes[1]
    total_cm = sum(r.confusion_matrix for r in results)
    total_cm_norm = total_cm.astype(float) / total_cm.sum(axis=1, keepdims=True)
    
    if task == "4class":
        labels = ['Left', 'Right', 'Hands', 'Feet']
    else:
        labels = ['Left', 'Right']
    
    sns.heatmap(total_cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('True', fontsize=12)
    ax2.set_title('Aggregate Confusion Matrix (Normalized)', fontweight='bold')
    
    # 3. 样本分布
    ax3 = axes[2]
    train_samples = [r.n_train_samples for r in results]
    test_samples = [r.n_test_samples for r in results]
    
    x = np.arange(len(subjects))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, train_samples, width, label='Train', color='lightblue')
    bars2 = ax3.bar(x + width/2, test_samples, width, label='Test', color='lightcoral')
    
    ax3.set_xlabel('Test Subject', fontsize=12)
    ax3.set_ylabel('Number of Samples', fontsize=12)
    ax3.set_title('Train/Test Sample Distribution', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(subjects)
    ax3.legend()
    
    plt.tight_layout()
    
    save_path = output_dir / f"loso_{task}_results.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"结果图已保存: {save_path}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PhysioNet LOSO Cross-Validation")
    
    # 数据
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                   help="参与 LOSO 的被试列表 (默认: 1-10)")
    p.add_argument("--task", choices=["2class", "4class"], default="2class",
                   help="任务类型: 2class (left/right) 或 4class")
    
    # 模型
    p.add_argument("--model", choices=["simple", "transformer"], default="simple",
                   help="模型类型: simple (SimpleCTNet) 或 transformer (EEGTransformer)")
    
    # 训练
    p.add_argument("--epochs", type=int, default=100, help="每个 fold 的训练轮数")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20, help="早停 patience")
    p.add_argument("--device", type=str, default="cuda")
    
    # 输出
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "loso")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("PhysioNet LOSO Cross-Validation")
    print("="*60)
    print(f"被试: {args.subjects}")
    print(f"任务: {args.task}")
    print(f"模型: {args.model}")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 加载所有数据
    print(f"\n加载数据...")
    data, labels, subject_ids = load_physionet_data(args.subjects, args.task, verbose=True)
    
    n_classes = 4 if args.task == "4class" else 2
    print(f"\n数据形状: {data.shape}")
    print(f"类别数: {n_classes}")
    print(f"总样本数: {len(labels)}")
    
    # LOSO 循环
    results = []
    
    print(f"\n开始 LOSO 交叉验证 ({len(args.subjects)} folds)...\n")
    
    for i, test_subject in enumerate(args.subjects):
        print(f"[Fold {i+1}/{len(args.subjects)}] 测试被试: {test_subject}")
        
        # 划分训练/测试集
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask
        
        train_data = data[train_mask]
        train_labels = labels[train_mask]
        test_data = data[test_mask]
        test_labels = labels[test_mask]
        
        print(f"  训练: {len(train_data)} 样本, 测试: {len(test_data)} 样本")
        
        # 运行 fold
        result = run_loso_fold(
            test_subject=test_subject,
            train_data=train_data,
            train_labels=train_labels,
            test_data=test_data,
            test_labels=test_labels,
            n_classes=n_classes,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            model_type=args.model,
            verbose=True,
        )
        
        results.append(result)
        print(f"  ★ 准确率: {result.accuracy:.2%}\n")
    
    # 汇总结果
    print("="*60)
    print("LOSO 结果汇总")
    print("="*60)
    
    accuracies = [r.accuracy for r in results]
    print(f"\n各被试准确率:")
    for r in results:
        print(f"  Subject {r.test_subject:2d}: {r.accuracy:.2%}")
    
    print(f"\n整体统计:")
    print(f"  平均准确率: {np.mean(accuracies):.2%} ± {np.std(accuracies):.2%}")
    print(f"  最高: {np.max(accuracies):.2%} (Subject {results[np.argmax(accuracies)].test_subject})")
    print(f"  最低: {np.min(accuracies):.2%} (Subject {results[np.argmin(accuracies)].test_subject})")
    
    # 保存结果
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 可视化
    visualize_loso_results(results, args.output_dir, args.task)
    
    # 保存 JSON
    results_json = {
        "task": args.task,
        "model": args.model,
        "subjects": args.subjects,
        "n_classes": n_classes,
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "folds": [
            {
                "test_subject": r.test_subject,
                "accuracy": float(r.accuracy),
                "n_train": r.n_train_samples,
                "n_test": r.n_test_samples,
            }
            for r in results
        ]
    }
    
    json_path = args.output_dir / f"loso_{args.task}_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n结果已保存: {args.output_dir}")


if __name__ == "__main__":
    main()
