#!/usr/bin/env python3
"""
EEG 分类准确率验证 — 使用真实 BCI IV-2a 数据 (无需硬件)

测试内容:
1. 加载 PhysioNet / BCI IV-2a .mat 数据
2. 创建4组偏向性数据集 (40/30/20/10 分布)
3. 通过 CTNet 分类并比较 predicted vs actual 分布
4. 证明模型能区分不同 MI 类型，不是随机猜测

用法:
    python scripts/test_eeg_classification.py
    python scripts/test_eeg_classification.py --subjects 1 2 3

创建时间: 2026-03-10
"""

from __future__ import annotations

import os
os.environ['MPLBACKEND'] = 'Agg'

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import scipy.io

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# 数据加载
# ============================================================================

CLASS_NAMES = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]

def load_subject_data(subject_id: int, data_dir: str = "mymat_raw") -> Tuple[np.ndarray, np.ndarray]:
    """加载单个被试的测试数据"""
    test_file = _ROOT / data_dir / f"A{subject_id:02d}E.mat"
    train_file = _ROOT / data_dir / f"A{subject_id:02d}T.mat"
    
    if not test_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {test_file}")
    
    mat = scipy.io.loadmat(str(test_file))
    data = mat['data']  # (N, 22, 1000)
    labels = mat['label'].flatten() - 1  # → 0-based
    
    # 归一化 (使用训练集统计)
    if train_file.exists():
        train_mat = scipy.io.loadmat(str(train_file))
        train_data = train_mat['data']
        mean = np.mean(train_data)
        std = np.std(train_data)
    else:
        mean = np.mean(data)
        std = np.std(data)
    
    data = (data - mean) / (std + 1e-8)
    data = np.expand_dims(data, axis=1)  # → (N, 1, 22, 1000)
    
    return data, labels


def create_biased_dataset(
    data: np.ndarray,
    labels: np.ndarray,
    class_proportions: Dict[int, float],
    total_samples: int = 100,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建偏向性数据集
    
    Args:
        data: (N, 1, 22, 1000)
        labels: (N,) 0-3
        class_proportions: {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        total_samples: 总样本数
    """
    rng = np.random.RandomState(seed)
    
    selected_data = []
    selected_labels = []
    
    for cls_id, proportion in class_proportions.items():
        n_samples = max(1, int(total_samples * proportion))
        cls_indices = np.where(labels == cls_id)[0]
        
        if len(cls_indices) == 0:
            continue
        
        chosen = rng.choice(cls_indices, size=n_samples, replace=True)
        selected_data.append(data[chosen])
        selected_labels.append(labels[chosen])
    
    return np.concatenate(selected_data), np.concatenate(selected_labels)


# ============================================================================
# 分类验证
# ============================================================================

def run_classification_test(
    model,
    data: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
    batch_size: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """运行 CTNet 分类"""
    model.eval()
    predictions = []
    probabilities = []
    
    data_tensor = torch.from_numpy(data).float()
    
    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i+batch_size].to(device)
            _, logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
    
    return np.array(predictions), np.array(probabilities)


# ============================================================================
# 可视化
# ============================================================================

def plot_biased_results(
    all_results: Dict,
    output_path: Path,
    subject_id: int,
):
    """绘制偏向性测试结果"""
    n_sets = len(all_results)
    fig, axes = plt.subplots(2, n_sets, figsize=(5 * n_sets, 8))
    
    fig.suptitle(f"CTNet Classification on Biased Real EEG Data (Subject {subject_id})\n"
                 f"Proving model discriminates — NOT random guessing",
                 fontsize=13, fontweight='bold')
    
    colors_actual = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    colors_pred = ['#1565C0', '#C62828', '#2E7D32', '#E65100']
    
    for col, (set_name, result) in enumerate(all_results.items()):
        actual_dist = result['actual_dist']
        pred_dist = result['pred_dist']
        accuracy = result['accuracy']
        
        # Top: distribution comparison
        ax = axes[0, col] if n_sets > 1 else axes[0]
        x = np.arange(4)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, actual_dist, width, label='Actual',
                       color=colors_actual, alpha=0.85, edgecolor='white')
        bars2 = ax.bar(x + width/2, pred_dist, width, label='Predicted',
                       color=colors_pred, alpha=0.65, edgecolor='white',
                       hatch='//')
        
        ax.set_xticks(x)
        ax.set_xticklabels(['Left', 'Right', 'Feet', 'Tongue'], fontsize=8)
        ax.set_ylabel('Proportion')
        ax.set_title(f'{set_name}\nAcc={accuracy:.1%}', fontweight='bold', fontsize=10)
        ax.legend(fontsize=7)
        ax.set_ylim(0, 0.55)
        ax.grid(True, alpha=0.2, axis='y')
        
        # Bottom: confusion-style heatmap
        ax2 = axes[1, col] if n_sets > 1 else axes[1]
        cm = result['confusion_matrix']
        im = ax2.imshow(cm, cmap='Blues', vmin=0)
        ax2.set_xticks(range(4))
        ax2.set_yticks(range(4))
        ax2.set_xticklabels(['L', 'R', 'F', 'T'], fontsize=9)
        ax2.set_yticklabels(['L', 'R', 'F', 'T'], fontsize=9)
        ax2.set_xlabel('Predicted', fontsize=9)
        ax2.set_ylabel('Actual', fontsize=9)
        
        for i in range(4):
            for j in range(4):
                ax2.text(j, i, f'{cm[i,j]}', ha='center', va='center',
                        fontsize=9, color='white' if cm[i,j] > cm.max()/2 else 'black')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主流程
# ============================================================================

def main():
    import argparse
    p = argparse.ArgumentParser(description="EEG 分类准确率验证")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3],
                   help="被试 ID (默认 1 2 3)")
    p.add_argument("--data-dir", type=str, default="mymat_raw",
                   help="数据目录")
    p.add_argument("--model", type=str, 
                   default=str(_ROOT / "A_heads_2_depth_6" / "model_1.pth"))
    p.add_argument("--total-samples", type=int, default=100,
                   help="每组数据集的总样本数")
    args = p.parse_args()
    
    output_dir = _ROOT / "outputs" / "eeg_classification_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EEG 分类准确率验证 — 偏向性真实数据测试")
    print("=" * 70)
    print(f"  数据: {args.data_dir}")
    print(f"  模型: {args.model}")
    print(f"  被试: {args.subjects}")
    print(f"  每组样本数: {args.total_samples}")
    
    # 加载 CTNet 模型
    import CTNet_model
    import __main__
    for attr_name in dir(CTNet_model):
        if not attr_name.startswith('_'):
            setattr(__main__, attr_name, getattr(CTNet_model, attr_name))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(args.model, map_location=device, weights_only=False)
    model.eval()
    print(f"\n[CTNet] ✓ 模型已加载 (设备: {device})")
    
    # 定义4组偏向性分布
    BIASED_SETS = {
        "Set A (40L/30R/20F/10T)": {0: 0.40, 1: 0.30, 2: 0.20, 3: 0.10},
        "Set B (10L/40R/30F/20T)": {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.10},
        "Set C (20L/10R/40F/30T)": {0: 0.20, 1: 0.10, 2: 0.40, 3: 0.30},
        "Set D (30L/20R/10F/40T)": {0: 0.30, 1: 0.20, 2: 0.10, 3: 0.40},
    }
    
    all_subject_results = {}
    
    for sid in args.subjects:
        print(f"\n{'='*60}")
        print(f"Subject {sid}")
        print(f"{'='*60}")
        
        try:
            data, labels = load_subject_data(sid, args.data_dir)
            print(f"  数据形状: {data.shape}, 标签分布: {np.bincount(labels, minlength=4)}")
        except FileNotFoundError as e:
            print(f"  ⚠ {e}")
            continue
        
        # === Baseline: 完整测试集 ===
        preds_full, probs_full = run_classification_test(model, data, labels, device)
        acc_full = np.mean(preds_full == labels)
        print(f"\n  [Baseline] 完整测试集准确率: {acc_full:.1%}")
        print(f"  真实标签分布: {np.bincount(labels, minlength=4)}")
        print(f"  预测分布:     {np.bincount(preds_full, minlength=4)}")
        
        # === 偏向性测试 ===
        subject_results = {}
        
        for set_name, proportions in BIASED_SETS.items():
            biased_data, biased_labels = create_biased_dataset(
                data, labels, proportions, args.total_samples,
                seed=42 + sid,
            )
            
            preds, probs = run_classification_test(model, biased_data, biased_labels, device)
            
            actual_counts = np.bincount(biased_labels, minlength=4)
            pred_counts = np.bincount(preds, minlength=4)
            acc = np.mean(preds == biased_labels)
            
            # 混淆矩阵
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(biased_labels, preds, labels=[0,1,2,3])
            
            subject_results[set_name] = {
                'actual_dist': actual_counts / actual_counts.sum(),
                'pred_dist': pred_counts / pred_counts.sum(),
                'accuracy': acc,
                'confusion_matrix': cm,
                'actual_counts': actual_counts.tolist(),
                'pred_counts': pred_counts.tolist(),
            }
            
            print(f"\n  [{set_name}]")
            print(f"    实际: {actual_counts} → 预测: {pred_counts}")
            print(f"    准确率: {acc:.1%}")
        
        # 保存图
        plot_biased_results(
            subject_results,
            output_dir / f"subject_{sid}_biased_test.png",
            sid,
        )
        print(f"\n  ✓ 图已保存: {output_dir}/subject_{sid}_biased_test.png")
        
        all_subject_results[sid] = {
            'baseline_accuracy': float(acc_full),
            'biased_tests': {
                name: {
                    'accuracy': float(r['accuracy']),
                    'actual_counts': r['actual_counts'],
                    'pred_counts': r['pred_counts'],
                }
                for name, r in subject_results.items()
            }
        }
    
    # ============================================================
    # 汇总
    # ============================================================
    print("\n" + "=" * 70)
    print("汇总结果")
    print("=" * 70)
    
    print(f"\n{'Subject':>10} | {'Baseline':>10} | {'Set A':>10} | {'Set B':>10} | {'Set C':>10} | {'Set D':>10}")
    print("-" * 70)
    
    set_names = list(BIASED_SETS.keys())
    for sid, results in all_subject_results.items():
        row = f"  S{sid:02d}      | {results['baseline_accuracy']:>9.1%} |"
        for sn in set_names:
            if sn in results['biased_tests']:
                row += f" {results['biased_tests'][sn]['accuracy']:>9.1%} |"
            else:
                row += f"      N/A |"
        print(row)
    
    mean_baseline = np.mean([r['baseline_accuracy'] for r in all_subject_results.values()])
    print(f"\n  平均 Baseline 准确率: {mean_baseline:.1%}")
    print(f"  随机猜测准确率:     25.0%")
    print(f"  模型提升: +{mean_baseline - 0.25:.1%}")
    
    if mean_baseline > 0.35:
        print("\n  ✅ 模型显著优于随机猜测 → 能够区分不同 MI 类型")
    else:
        print("\n  ⚠ 模型准确率较低，可能需要进一步优化")
    
    # 保存 JSON 结果
    json_path = output_dir / "classification_results.json"
    with open(json_path, 'w') as f:
        json.dump(all_subject_results, f, indent=2)
    print(f"\n  结果 JSON: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
