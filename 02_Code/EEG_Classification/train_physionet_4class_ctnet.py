#!/usr/bin/env python3
"""
PhysioNet 4 类分类 - 使用完整 CTNet (EEGTransformer)

任务映射:
    - 0: 左手想象 → left
    - 1: 右手想象 → right  
    - 2: 双手想象 → up
    - 3: 双脚想象 → down

创建时间: 2026-02-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 项目根目录
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

# 导入完整 CTNet
from CTNet_model import (
    PatchEmbeddingCNN,
    MultiHeadAttention, 
    TransformerEncoder,
    ClassificationHead,
)

# 导入 PhysioNet 加载器
from scripts.physionet_loader import (
    load_subject_4class_mne,
    load_multiple_subjects_4class_mne,
    preprocess_data,
    MNE_AVAILABLE,
)


# ============================================================================
# 完整 CTNet (EEGTransformer)
# ============================================================================

class EEGTransformer(nn.Module):
    """
    完整 CTNet 架构
    
    结构:
        1. PatchEmbeddingCNN: 时空特征提取
        2. TransformerEncoder: 自注意力建模
        3. ClassificationHead: 分类输出
    """
    
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
        
        # 1. Patch Embedding (CNN)
        self.patch_embed = PatchEmbeddingCNN(
            n_channels=n_channels,
            d_model=d_model,
            patch_size=patch_size,
        )
        
        # 计算序列长度
        self.seq_len = n_times // patch_size
        
        # 2. Positional Encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.seq_len, d_model) * 0.02)
        
        # 3. Transformer Encoder
        self.transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        
        # 4. Classification Head
        self.classifier = ClassificationHead(
            d_model=d_model,
            n_classes=n_classes,
            dropout=dropout,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, channels, times) 或 (batch, channels, times)
        Returns:
            logits: (batch, n_classes)
        """
        # 确保输入格式正确
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, C, T) -> (B, 1, C, T)
        
        # 1. Patch Embedding
        x = self.patch_embed(x)  # (B, seq_len, d_model)
        
        # 2. Add Positional Encoding
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x)
        
        # 3. Transformer Encoder
        x = self.transformer(x)  # (B, seq_len, d_model)
        
        # 4. Classification (使用 [CLS] token 或全局平均)
        x = x.mean(dim=1)  # Global Average Pooling
        logits = self.classifier(x)
        
        return logits


# ============================================================================
# 训练函数
# ============================================================================

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
        
        # 梯度裁剪
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
) -> Tuple[float, float]:
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
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
    
    return total_loss / total, correct / total


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 200,
    lr: float = 1e-3,
    patience: int = 30,
) -> Tuple[nn.Module, float, list, list]:
    """
    训练和评估模型
    
    Returns:
        model: 最佳模型
        best_acc: 最佳准确率
        train_accs: 训练准确率历史
        test_accs: 测试准确率历史
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    best_weights = None
    epochs_no_improve = 0
    
    train_accs = []
    test_accs = []
    
    print(f"\n训练 {epochs} 轮 (早停: patience={patience})...")
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        # 早停检查
        if test_acc > best_acc:
            best_acc = test_acc
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # 打印进度
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: Train {train_acc:.2%}, Test {test_acc:.2%} (best: {best_acc:.2%} @ ep{epoch+1-epochs_no_improve})")
        
        # 早停
        if epochs_no_improve >= patience:
            print(f"  ⏹ 早停: 连续 {patience} 个 epoch 无改善")
            print(f"    最佳: Epoch {epoch+1-patience}, 准确率 {best_acc:.2%}")
            break
    
    # 恢复最佳权重
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print(f"  ★ 已恢复最佳模型: {best_acc:.2%}")
    
    return model, best_acc, train_accs, test_accs


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="PhysioNet 4类 CTNet 训练")
    
    # 数据
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 11)),
                   help="被试编号 (默认: 1-10)")
    p.add_argument("--mat-file", type=Path, default=None,
                   help="预处理好的 .mat 文件路径 (如果有)")
    
    # 模型
    p.add_argument("--d-model", type=int, default=64, help="Transformer 维度")
    p.add_argument("--n-heads", type=int, default=8, help="注意力头数")
    p.add_argument("--n-layers", type=int, default=2, help="Transformer 层数")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout 率")
    
    # 训练
    p.add_argument("--epochs", type=int, default=200, help="训练轮数")
    p.add_argument("--batch-size", type=int, default=32, help="批次大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--patience", type=int, default=30, help="早停 patience")
    p.add_argument("--device", type=str, default="cuda", help="设备")
    
    # 输出
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "physionet_4class_ctnet",
                   help="输出目录")
    p.add_argument("--filter", action="store_true", help="应用 8-30 Hz 带通滤波")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("PhysioNet 4 类 CTNet 训练")
    print("="*60)
    print(f"标签映射: 0=left, 1=right, 2=hands(up), 3=feet(down)")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    
    # 加载数据
    if args.mat_file and args.mat_file.exists():
        print(f"\n从 .mat 文件加载: {args.mat_file}")
        from scipy.io import loadmat
        mat = loadmat(str(args.mat_file))
        data = mat["data"]
        labels = mat["label"].flatten()
        subject_ids = mat.get("subject_id", np.zeros(len(labels))).flatten()
    else:
        if not MNE_AVAILABLE:
            print("❌ 需要安装 MNE: pip install mne")
            return
        
        print(f"\n加载 4 类数据 (被试: {args.subjects})...")
        data, labels, subject_ids = load_multiple_subjects_4class_mne(
            args.subjects, verbose=True
        )
    
    print(f"\n数据形状: {data.shape}")
    print(f"标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 滤波
    if args.filter:
        print("\n应用 8-30 Hz 带通滤波...")
        data = preprocess_data(data)
    
    # 标准化
    mean, std = data.mean(), data.std()
    data = (data - mean) / (std + 1e-8)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    print(f"\n训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 创建 DataLoader
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # 创建模型
    n_channels = data.shape[1]
    n_times = data.shape[2]
    n_classes = 4
    
    print(f"\n模型配置:")
    print(f"  输入: {n_channels} channels × {n_times} times")
    print(f"  输出: {n_classes} classes")
    print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}, n_layers: {args.n_layers}")
    
    model = EEGTransformer(
        n_channels=n_channels,
        n_times=n_times,
        n_classes=n_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    
    # 计算参数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {n_params:,}")
    
    # 训练
    model, best_acc, train_accs, test_accs = train_and_evaluate(
        model, train_loader, test_loader, device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
    )
    
    # 保存模型
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "physionet_4class_ctnet.pth"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_channels': n_channels,
        'n_times': n_times,
        'n_classes': n_classes,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'accuracy': best_acc,
        'subjects': args.subjects,
    }, model_path)
    
    print(f"\n✅ 训练完成!")
    print(f"  最佳准确率: {best_acc:.2%}")
    print(f"  模型已保存: {model_path}")
    
    # 保存训练曲线数据
    history_path = args.output_dir / "training_history.npz"
    np.savez(history_path, train_accs=train_accs, test_accs=test_accs)
    print(f"  训练历史: {history_path}")


if __name__ == "__main__":
    main()



