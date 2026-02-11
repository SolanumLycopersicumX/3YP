#!/usr/bin/env python3
"""
DQN 训练结果可视化

生成:
1. 奖励曲线
2. 到达率曲线
3. 损失曲线
4. Epsilon 衰减曲线

用法:
  python scripts/plot_dqn_training.py --log outputs/dqn_rl_policy.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def moving_average(data: list, window: int = 50) -> np.ndarray:
    """计算移动平均"""
    if len(data) < window:
        return np.array(data)
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


def plot_training_curves(logs: dict, save_path: Path, window: int = 50):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = logs["episode"]
    
    # 1. 奖励曲线
    ax = axes[0, 0]
    rewards = logs["reward"]
    ax.plot(episodes, rewards, alpha=0.3, color="blue", label="原始")
    if len(rewards) >= window:
        ma_rewards = moving_average(rewards, window)
        ma_eps = episodes[window-1:]
        ax.plot(ma_eps, ma_rewards, color="blue", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("训练奖励")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 到达率曲线
    ax = axes[0, 1]
    reached = logs["reached"]
    if len(reached) >= window:
        ma_reached = moving_average(reached, window) * 100
        ma_eps = episodes[window-1:]
        ax.plot(ma_eps, ma_reached, color="green", linewidth=2)
    else:
        ax.plot(episodes, np.array(reached) * 100, color="green")
    ax.set_xlabel("Episode")
    ax.set_ylabel("到达率 (%)")
    ax.set_title(f"目标到达率 (MA-{window})")
    ax.set_ylim([0, 105])
    ax.grid(alpha=0.3)
    ax.axhline(y=80, color="red", linestyle="--", alpha=0.5, label="80% 基准")
    ax.legend()
    
    # 3. 损失曲线
    ax = axes[1, 0]
    losses = logs["loss"]
    non_zero_losses = [(e, l) for e, l in zip(episodes, losses) if l > 0]
    if non_zero_losses:
        loss_eps, loss_vals = zip(*non_zero_losses)
        ax.plot(loss_eps, loss_vals, alpha=0.5, color="red")
        if len(loss_vals) >= window:
            ma_loss = moving_average(list(loss_vals), window)
            ax.plot(loss_eps[window-1:], ma_loss, color="darkred", linewidth=2, label=f"MA({window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Loss")
    ax.set_title("训练损失")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. Epsilon 曲线
    ax = axes[1, 1]
    epsilons = logs["epsilon"]
    ax.plot(episodes, epsilons, color="purple", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("探索率 ε 衰减")
    ax.grid(alpha=0.3)
    ax.axhline(y=0.1, color="orange", linestyle="--", alpha=0.5, label="ε=0.1")
    ax.legend()
    
    fig.suptitle("DQN 强化学习训练过程", fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"保存训练曲线到: {save_path}")


def print_summary(logs: dict):
    """打印训练摘要"""
    print("\n" + "="*60)
    print("训练摘要")
    print("="*60)
    
    rewards = logs["reward"]
    reached = logs["reached"]
    steps = logs["steps"]
    
    total_eps = len(rewards)
    
    # 分阶段统计
    stages = [
        ("前期 (1-100)", 0, min(100, total_eps)),
        ("中期 (101-500)", 100, min(500, total_eps)),
        ("后期 (501-end)", 500, total_eps),
    ]
    
    print(f"\n总训练 Episodes: {total_eps}")
    print("\n阶段统计:")
    print("-" * 60)
    print(f"{'阶段':<20} {'平均奖励':>12} {'到达率':>12} {'平均步数':>12}")
    print("-" * 60)
    
    for name, start, end in stages:
        if start >= end:
            continue
        avg_reward = np.mean(rewards[start:end])
        reach_rate = np.mean(reached[start:end]) * 100
        avg_steps = np.mean(steps[start:end])
        print(f"{name:<20} {avg_reward:>12.2f} {reach_rate:>11.1f}% {avg_steps:>12.1f}")
    
    print("-" * 60)
    
    # 最佳性能
    best_reward_idx = np.argmax(rewards)
    print(f"\n最高奖励: {rewards[best_reward_idx]:.2f} (Episode {best_reward_idx + 1})")
    print(f"最终到达率 (最后100ep): {np.mean(reached[-100:])*100:.1f}%")


def parse_args():
    p = argparse.ArgumentParser(description="可视化 DQN 训练结果")
    p.add_argument("--log", type=Path, required=True, help="训练日志 JSON 文件")
    p.add_argument("--output", type=Path, default=None, help="输出图像路径")
    p.add_argument("--window", type=int, default=50, help="移动平均窗口大小")
    return p.parse_args()


def main():
    args = parse_args()
    
    # 加载日志
    with open(args.log, "r", encoding="utf-8") as f:
        logs = json.load(f)
    
    # 确定输出路径
    if args.output is None:
        args.output = args.log.with_suffix(".png")
    
    # 绘图
    plot_training_curves(logs, args.output, args.window)
    
    # 打印摘要
    print_summary(logs)


if __name__ == "__main__":
    main()



