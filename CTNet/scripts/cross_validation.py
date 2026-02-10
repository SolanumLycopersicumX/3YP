#!/usr/bin/env python3
"""
K-Fold 交叉验证与多种子测试模块

解决教授反馈的问题:
1. 添加验证集划分 (Train/Val/Test)
2. 实现 K-Fold 交叉验证
3. 多随机种子测试，报告 mean ± std

用法示例:
    python scripts/cross_validation.py --model dqn --dataset A --subject 1 --k-folds 5

创建时间: 2026-02-03
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# 数据划分策略
# ============================================================================

@dataclass
class DataSplit:
    """数据划分结果"""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    X_test: np.ndarray = field(default_factory=lambda: np.array([]))
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def summary(self) -> str:
        lines = [
            f"Train: {len(self.y_train)} samples",
        ]
        if self.X_val is not None:
            lines.append(f"Val:   {len(self.y_val)} samples")
        if len(self.y_test) > 0:
            lines.append(f"Test:  {len(self.y_test)} samples")
        return "\n".join(lines)


def split_train_val_test(
    X: np.ndarray,
    y: np.ndarray,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> DataSplit:
    """
    将数据划分为训练集、验证集和测试集
    
    Parameters
    ----------
    X : 特征数据 [N, ...]
    y : 标签 [N]
    val_ratio : 验证集比例
    test_ratio : 测试集比例
    random_state : 随机种子
    stratify : 是否按类别分层
    
    Returns
    -------
    DataSplit 对象
    """
    stratify_arr = y if stratify else None
    
    # 先分出测试集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=stratify_arr
    )
    
    # 再从剩余数据中分出验证集
    val_size = val_ratio / (1 - test_ratio)  # 调整比例
    stratify_temp = y_temp if stratify else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=stratify_temp
    )
    
    return DataSplit(
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
    )


def get_kfold_splits(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    生成 K-Fold 交叉验证的数据划分
    
    Returns
    -------
    List of (X_train, y_train, X_val, y_val) tuples
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        splits.append((X_train, y_train, X_val, y_val))
    
    return splits


# ============================================================================
# 交叉验证训练器
# ============================================================================

@dataclass
class CVResult:
    """交叉验证结果"""
    fold_results: List[Dict[str, float]]
    mean_accuracy: float
    std_accuracy: float
    all_seeds_results: Optional[List[Dict]] = None
    
    def summary(self) -> str:
        lines = [
            "=" * 50,
            "K-Fold Cross Validation Results",
            "=" * 50,
        ]
        for i, fold in enumerate(self.fold_results):
            lines.append(f"Fold {i+1}: Accuracy = {fold['accuracy']:.2%}")
        lines.append("-" * 50)
        lines.append(f"Mean: {self.mean_accuracy:.2%} ± {self.std_accuracy:.2%}")
        lines.append("=" * 50)
        return "\n".join(lines)


def train_and_evaluate(
    model_fn,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = False,
) -> Dict[str, float]:
    """
    训练模型并在验证集上评估
    
    Parameters
    ----------
    model_fn : 创建模型的函数 (返回 nn.Module)
    X_train, y_train : 训练数据
    X_val, y_val : 验证数据
    
    Returns
    -------
    Dict with 'accuracy', 'loss', etc.
    """
    # 准备数据
    def prep(x):
        x = np.asarray(x, dtype=np.float32)
        mean, std = x.mean(), (x.std() or 1.0)
        x = (x - mean) / std
        # 转换为 (batch, seq_len, state_dim)
        if x.ndim == 3:  # [N, Ch, T]
            x = np.transpose(x, (0, 2, 1))
        return torch.tensor(x, dtype=torch.float32)
    
    X_tr = prep(X_train).to(device)
    y_tr = torch.tensor(y_train.reshape(-1), dtype=torch.long).to(device)
    X_va = prep(X_val).to(device)
    y_va = torch.tensor(y_val.reshape(-1), dtype=torch.long).to(device)
    
    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # 创建模型
    model = model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        out = model(X_va)
        pred = out.argmax(dim=1)
        accuracy = (pred == y_va).float().mean().item()
        val_loss = criterion(out, y_va).item()
    
    return {
        "accuracy": accuracy,
        "loss": val_loss,
    }


def run_kfold_cv(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> CVResult:
    """
    执行 K-Fold 交叉验证
    """
    splits = get_kfold_splits(X, y, n_splits=n_splits, random_state=random_state)
    
    fold_results = []
    for i, (X_train, y_train, X_val, y_val) in enumerate(splits):
        if verbose:
            print(f"\n[Fold {i+1}/{n_splits}] Training...")
        
        result = train_and_evaluate(
            model_fn, X_train, y_train, X_val, y_val,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device, verbose=False
        )
        fold_results.append(result)
        
        if verbose:
            print(f"[Fold {i+1}/{n_splits}] Accuracy: {result['accuracy']:.2%}")
    
    accuracies = [r["accuracy"] for r in fold_results]
    
    return CVResult(
        fold_results=fold_results,
        mean_accuracy=np.mean(accuracies),
        std_accuracy=np.std(accuracies),
    )


def run_multi_seed_test(
    model_fn,
    X: np.ndarray,
    y: np.ndarray,
    *,
    seeds: List[int] = [42, 123, 456, 789, 1024],
    test_ratio: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> CVResult:
    """
    多随机种子测试，验证结果稳定性
    """
    seed_results = []
    
    for seed in seeds:
        if verbose:
            print(f"\n[Seed {seed}] Training...")
        
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 划分数据
        split = split_train_val_test(X, y, val_ratio=0.1, test_ratio=test_ratio, random_state=seed)
        
        result = train_and_evaluate(
            model_fn, split.X_train, split.y_train, split.X_test, split.y_test,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device, verbose=False
        )
        result["seed"] = seed
        seed_results.append(result)
        
        if verbose:
            print(f"[Seed {seed}] Accuracy: {result['accuracy']:.2%}")
    
    accuracies = [r["accuracy"] for r in seed_results]
    
    return CVResult(
        fold_results=seed_results,
        mean_accuracy=np.mean(accuracies),
        std_accuracy=np.std(accuracies),
    )


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="K-Fold 交叉验证与多种子测试")
    
    p.add_argument("--dataset", choices=["A", "B"], default="A",
                   help="数据集: A=IV-2a (22ch, 4class), B=IV-2b (3ch, 2class)")
    p.add_argument("--subject", type=int, default=1, help="被试编号 (1-9)")
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    
    p.add_argument("--mode", choices=["kfold", "multi-seed", "both"], default="both",
                   help="验证模式")
    p.add_argument("--k-folds", type=int, default=5, help="K-Fold 的折数")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 1024],
                   help="多种子测试的种子列表")
    
    p.add_argument("--model", choices=["dqn", "transformer"], default="dqn",
                   help="模型类型")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    
    p.add_argument("--output", type=Path, default=Path("outputs/cv_results.json"))
    
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    from gym_control import load_data_evaluate, number_class_channel
    
    X_train, y_train, X_test, y_test = load_data_evaluate(
        args.data_dir, args.dataset, args.subject, "subject-dependent"
    )
    
    # 合并训练和测试集，然后重新划分
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).reshape(-1)
    
    # 标签从 1 开始，转换为从 0 开始 (CrossEntropyLoss 要求)
    if y.min() == 1:
        y = y - 1
        print(f"Labels converted from 1-based to 0-based: {np.unique(y)}")
    
    n_classes, n_channels = number_class_channel(args.dataset)
    seq_len = X.shape[-1]
    state_dim = n_channels
    
    print(f"\nDataset: {args.dataset} (Subject {args.subject})")
    print(f"Total samples: {len(y)}")
    print(f"Classes: {n_classes}, Channels: {n_channels}, Seq length: {seq_len}")
    
    # 模型工厂
    if args.model == "dqn":
        from scripts.dqn_model import DQNNetwork
        def model_fn():
            return DQNNetwork(
                state_dim=state_dim,
                action_dim=n_classes,
                seq_len=seq_len,
            )
    else:
        from scripts.dqn_transformer import TransformerDQN
        def model_fn():
            return TransformerDQN(
                state_dim=state_dim,
                action_dim=n_classes,
                seq_len=seq_len,
            )
    
    results = {}
    
    # K-Fold 交叉验证
    if args.mode in ["kfold", "both"]:
        print(f"\n{'='*50}")
        print(f"Running {args.k_folds}-Fold Cross Validation")
        print(f"{'='*50}")
        
        cv_result = run_kfold_cv(
            model_fn, X, y,
            n_splits=args.k_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        
        print(cv_result.summary())
        results["kfold"] = {
            "n_splits": args.k_folds,
            "fold_results": cv_result.fold_results,
            "mean": cv_result.mean_accuracy,
            "std": cv_result.std_accuracy,
        }
    
    # 多种子测试
    if args.mode in ["multi-seed", "both"]:
        print(f"\n{'='*50}")
        print(f"Running Multi-Seed Test ({len(args.seeds)} seeds)")
        print(f"{'='*50}")
        
        seed_result = run_multi_seed_test(
            model_fn, X, y,
            seeds=args.seeds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
        )
        
        print(seed_result.summary())
        results["multi_seed"] = {
            "seeds": args.seeds,
            "seed_results": seed_result.fold_results,
            "mean": seed_result.mean_accuracy,
            "std": seed_result.std_accuracy,
        }
    
    # 保存结果
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    
    # 总结
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    if "kfold" in results:
        print(f"K-Fold CV:     {results['kfold']['mean']:.2%} ± {results['kfold']['std']:.2%}")
    if "multi_seed" in results:
        print(f"Multi-Seed:    {results['multi_seed']['mean']:.2%} ± {results['multi_seed']['std']:.2%}")


if __name__ == "__main__":
    main()

