#!/usr/bin/env python3
"""
CTNet 双数据集对比测试脚本

测试 IV-2a 和 IV-2b 数据集，可视化并对比结果

用法:
    python scripts/compare_datasets.py --epochs 100 --device cuda

创建时间: 2026-02-10
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

# 导入 CTNet
from CTNet_model import ExP as CTNetTrainer


def run_ctnet_test(
    dataset: str,
    subjects: List[int],
    data_dir: Path,
    output_dir: Path,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[int, Dict]:
    """
    对指定数据集的所有被试运行 CTNet 测试
    
    Returns
    -------
    Dict[subject_id, {'accuracy': float, 'best_epoch': int, ...}]
    """
    results = {}
    
    for subject in subjects:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Dataset {dataset} - Subject {subject}/{len(subjects)}")
            print(f"{'='*60}")
        
        result_name = str(output_dir / f"dataset_{dataset}")
        Path(result_name).mkdir(parents=True, exist_ok=True)
        
        try:
            trainer = CTNetTrainer(
                nsub=subject,
                data_dir=str(data_dir) + "/",
                result_name=result_name,
                dataset_type=dataset,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=lr,
                evaluate_mode="subject-dependent",
                validate_ratio=0.1,
                early_stopping=True,
                patience=20,
                verbose=verbose,
                plot_training=False,  # 不显示实时图
            )
            
            acc, test_label, y_pred, df_process, best_epoch = trainer.train()
            
            results[subject] = {
                'accuracy': acc,
                'best_epoch': best_epoch,
                'test_samples': len(test_label),
            }
            
            if verbose:
                print(f"Subject {subject}: Accuracy = {acc*100:.2f}%")
                
        except Exception as e:
            print(f"Subject {subject} 失败: {e}")
            results[subject] = {
                'accuracy': 0.0,
                'best_epoch': 0,
                'error': str(e),
            }
    
    return results


def visualize_comparison(
    results_a: Dict[int, Dict],
    results_b: Dict[int, Dict],
    output_path: Path,
):
    """
    可视化两个数据集的对比结果
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 准备数据
    subjects = list(results_a.keys())
    acc_a = [results_a[s]['accuracy'] * 100 for s in subjects]
    acc_b = [results_b[s]['accuracy'] * 100 for s in subjects]
    
    # 1. 柱状图对比
    ax1 = axes[0, 0]
    x = np.arange(len(subjects))
    width = 0.35
    bars1 = ax1.bar(x - width/2, acc_a, width, label='IV-2a (22ch, 4class)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, acc_b, width, label='IV-2b (3ch, 2class)', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Subject', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Per-Subject Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'S{s}' for s in subjects])
    ax1.legend(loc='lower right')
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bar, acc in zip(bars1, acc_a):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    for bar, acc in zip(bars2, acc_b):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}', ha='center', va='bottom', fontsize=8)
    
    # 添加随机基线
    ax1.axhline(y=25, color='#3498db', linestyle='--', alpha=0.5, label='IV-2a Chance (25%)')
    ax1.axhline(y=50, color='#e74c3c', linestyle='--', alpha=0.5, label='IV-2b Chance (50%)')
    
    # 2. 箱线图
    ax2 = axes[0, 1]
    bp = ax2.boxplot([acc_a, acc_b], labels=['IV-2a\n(22ch, 4class)', 'IV-2b\n(3ch, 2class)'],
                      patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 添加均值和标准差
    mean_a, std_a = np.mean(acc_a), np.std(acc_a)
    mean_b, std_b = np.mean(acc_b), np.std(acc_b)
    ax2.text(1, mean_a + 2, f'Mean: {mean_a:.1f}±{std_a:.1f}%', ha='center', fontsize=10)
    ax2.text(2, mean_b + 2, f'Mean: {mean_b:.1f}±{std_b:.1f}%', ha='center', fontsize=10)
    
    # 3. 散点图 - 被试间对比
    ax3 = axes[1, 0]
    ax3.scatter(acc_a, acc_b, s=100, c=subjects, cmap='viridis', alpha=0.8, edgecolors='black')
    for i, s in enumerate(subjects):
        ax3.annotate(f'S{s}', (acc_a[i], acc_b[i]), textcoords="offset points", 
                    xytext=(5, 5), fontsize=9)
    
    # 添加对角线
    max_val = max(max(acc_a), max(acc_b)) + 5
    min_val = min(min(acc_a), min(acc_b)) - 5
    ax3.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='y=x')
    
    ax3.set_xlabel('IV-2a Accuracy (%)', fontsize=12)
    ax3.set_ylabel('IV-2b Accuracy (%)', fontsize=12)
    ax3.set_title('Cross-Dataset Subject Performance', fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. 总结表格
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算相对于随机基线的提升
    improvement_a = mean_a - 25  # IV-2a 随机基线 25%
    improvement_b = mean_b - 50  # IV-2b 随机基线 50%
    
    summary_data = [
        ['Metric', 'IV-2a', 'IV-2b'],
        ['Channels', '22', '3'],
        ['Classes', '4', '2'],
        ['Random Baseline', '25%', '50%'],
        ['Mean Accuracy', f'{mean_a:.2f}%', f'{mean_b:.2f}%'],
        ['Std Deviation', f'±{std_a:.2f}%', f'±{std_b:.2f}%'],
        ['Best Subject', f'S{subjects[np.argmax(acc_a)]} ({max(acc_a):.1f}%)', 
                         f'S{subjects[np.argmax(acc_b)]} ({max(acc_b):.1f}%)'],
        ['Worst Subject', f'S{subjects[np.argmin(acc_a)]} ({min(acc_a):.1f}%)', 
                          f'S{subjects[np.argmin(acc_b)]} ({min(acc_b):.1f}%)'],
        ['Above Baseline', f'+{improvement_a:.1f}%', f'+{improvement_b:.1f}%'],
    ]
    
    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.325, 0.325])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # 设置表头样式
    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # 设置交替行颜色
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化结果已保存: {output_path}")
    
    return {
        'IV-2a': {'mean': mean_a, 'std': std_a, 'per_subject': acc_a},
        'IV-2b': {'mean': mean_b, 'std': std_b, 'per_subject': acc_b},
    }


def parse_args():
    p = argparse.ArgumentParser(description="CTNet 双数据集对比测试")
    
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/dataset_comparison/"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)),
                   help="要测试的被试列表 (默认 1-9)")
    p.add_argument("--dataset", choices=["A", "B", "both"], default="both",
                   help="测试哪个数据集")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("CTNet 双数据集对比测试")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"被试: {args.subjects}")
    print("="*60)
    
    results_a = {}
    results_b = {}
    
    # 测试 IV-2a
    if args.dataset in ["A", "both"]:
        print("\n" + "="*60)
        print("开始测试 IV-2a (22 channels, 4 classes)")
        print("="*60)
        
        start_time = time.time()
        results_a = run_ctnet_test(
            dataset="A",
            subjects=args.subjects,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        time_a = time.time() - start_time
        
        # 保存结果
        with open(args.output_dir / "results_2a.json", "w") as f:
            json.dump(results_a, f, indent=2)
        
        acc_list = [r['accuracy'] * 100 for r in results_a.values() if 'accuracy' in r]
        print(f"\nIV-2a 完成! 平均准确率: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")
        print(f"耗时: {time_a/60:.1f} 分钟")
    
    # 测试 IV-2b
    if args.dataset in ["B", "both"]:
        print("\n" + "="*60)
        print("开始测试 IV-2b (3 channels, 2 classes)")
        print("="*60)
        
        start_time = time.time()
        results_b = run_ctnet_test(
            dataset="B",
            subjects=args.subjects,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
        time_b = time.time() - start_time
        
        # 保存结果
        with open(args.output_dir / "results_2b.json", "w") as f:
            json.dump(results_b, f, indent=2)
        
        acc_list = [r['accuracy'] * 100 for r in results_b.values() if 'accuracy' in r]
        print(f"\nIV-2b 完成! 平均准确率: {np.mean(acc_list):.2f}% ± {np.std(acc_list):.2f}%")
        print(f"耗时: {time_b/60:.1f} 分钟")
    
    # 可视化对比
    if args.dataset == "both" and results_a and results_b:
        print("\n" + "="*60)
        print("生成对比可视化...")
        print("="*60)
        
        summary = visualize_comparison(
            results_a, results_b,
            args.output_dir / "dataset_comparison.png"
        )
        
        # 保存汇总
        with open(args.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # 打印最终总结
        print("\n" + "="*60)
        print("最终结果总结")
        print("="*60)
        print(f"{'数据集':<15} {'通道':<10} {'类别':<10} {'基线':<10} {'准确率':<20}")
        print("-"*60)
        print(f"{'IV-2a':<15} {'22':<10} {'4':<10} {'25%':<10} "
              f"{summary['IV-2a']['mean']:.2f}% ± {summary['IV-2a']['std']:.2f}%")
        print(f"{'IV-2b':<15} {'3':<10} {'2':<10} {'50%':<10} "
              f"{summary['IV-2b']['mean']:.2f}% ± {summary['IV-2b']['std']:.2f}%")
        print("="*60)
        
        # 相对提升
        improve_a = summary['IV-2a']['mean'] - 25
        improve_b = summary['IV-2b']['mean'] - 50
        print(f"\n相对于随机基线的提升:")
        print(f"  IV-2a: +{improve_a:.1f}% (25% → {summary['IV-2a']['mean']:.1f}%)")
        print(f"  IV-2b: +{improve_b:.1f}% (50% → {summary['IV-2b']['mean']:.1f}%)")


if __name__ == "__main__":
    main()
