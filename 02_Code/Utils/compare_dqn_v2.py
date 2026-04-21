#!/usr/bin/env python3
"""
DQN 网络架构对比实验 V2 - 改进版

修复 V1 中发现的训练崩溃问题：
1. 更慢的 ε 衰减
2. 添加 Gradient Clipping
3. 更频繁的 Target Network 更新
4. 添加 Double DQN
5. 添加 Prioritized Experience Replay (简化版)

用法:
  python scripts/compare_dqn_v2.py --episodes 2000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dqn_model import DQNNetwork, ReplayBuffer
from scripts.dqn_transformer import TransformerDQN, LightTransformerDQN
from scripts.train_dqn_rl import RLArmConfig, RLArm2DEnv


# ============================================================================
# 改进的训练组件
# ============================================================================

def epsilon_greedy_action_v2(q_values: torch.Tensor, epsilon: float) -> int:
    """ε-greedy 动作选择"""
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    return int(torch.argmax(q_values).item())


def double_dqn_training_step(
    policy_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple,
    *,
    gamma: float = 0.99,
    grad_clip: float = 1.0,
) -> float:
    """
    Double DQN 训练步骤
    
    改进：使用 policy_net 选择动作，target_net 评估 Q 值
    这减少了 Q 值过高估计的问题
    """
    states, actions, rewards, next_states, dones = batch
    
    # 当前 Q 值
    q_values = policy_net(states)
    q_values = q_values.gather(1, actions.view(-1, 1)).squeeze(1)
    
    with torch.no_grad():
        # Double DQN: 用 policy_net 选动作
        next_actions = policy_net(next_states).argmax(dim=1, keepdim=True)
        # 用 target_net 评估 Q 值
        next_q = target_net(next_states).gather(1, next_actions).squeeze(1)
        target = rewards + (1 - dones) * gamma * next_q
    
    # Huber Loss
    loss = F.smooth_l1_loss(q_values, target, reduction="mean")
    
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient Clipping
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=grad_clip)
    
    optimizer.step()
    
    return loss.item()


def linear_schedule(start: float, end: float, current: int, total: int) -> float:
    """线性调度器"""
    progress = min(1.0, current / total)
    return start + (end - start) * progress


def exponential_schedule(start: float, end: float, decay: float, current: int) -> float:
    """指数衰减调度器"""
    return max(end, start * (decay ** current))


# ============================================================================
# 改进的训练函数
# ============================================================================

def train_network_v2(
    network_name: str,
    network: nn.Module,
    env: RLArm2DEnv,
    *,
    num_episodes: int = 2000,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 1500,  # ε 衰减到最小值所需 episodes
    target_update_freq: int = 5,         # 更频繁更新 target
    soft_update_tau: float = 0.005,      # 软更新系数
    use_soft_update: bool = True,
    min_buffer_size: int = 1000,
    buffer_size: int = 100000,
    lr: float = 3e-4,                    # 较小学习率
    grad_clip: float = 1.0,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 100,
) -> Dict[str, Any]:
    """
    改进的训练函数 V2
    """
    print(f"\n{'='*60}")
    print(f"训练: {network_name}")
    print(f"参数量: {sum(p.numel() for p in network.parameters() if p.requires_grad):,}")
    print(f"改进: Double DQN, Soft Update (τ={soft_update_tau}), Slower ε decay")
    print(f"{'='*60}")
    
    state_dim = env.observation_dim
    
    # 创建 target network (使用相同配置)
    import copy
    target_net = copy.deepcopy(network).to(device)
    target_net.load_state_dict(network.state_dict())
    target_net.eval()
    
    # 优化器
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_episodes, eta_min=lr * 0.1
    )
    
    # Replay Buffer
    replay_buffer = ReplayBuffer(
        capacity=buffer_size,
        state_shape=(1, state_dim),
        device=device,
    )
    
    # 训练记录
    logs = {
        "episode": [],
        "reward": [],
        "steps": [],
        "epsilon": [],
        "loss": [],
        "reached": [],
        "lr": [],
    }
    
    start_time = time.time()
    best_reach_rate = 0.0
    best_weights = None
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        reached = False
        
        # 线性 ε 衰减
        epsilon = linear_schedule(
            epsilon_start, epsilon_end, ep, epsilon_decay_episodes
        )
        
        done = False
        while not done:
            # 动作选择
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(1)
            
            with torch.no_grad():
                q_values = network(state_tensor)
            action = epsilon_greedy_action_v2(q_values, epsilon)
            
            # 环境交互
            next_obs, reward, done, info = env.step(action)
            
            # 存储经验
            replay_buffer.add(
                obs.reshape(1, -1), action, reward,
                next_obs.reshape(1, -1), float(done)
            )
            
            # 训练
            if replay_buffer.size >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                loss = double_dqn_training_step(
                    network, target_net, optimizer, batch,
                    gamma=gamma, grad_clip=grad_clip
                )
                episode_losses.append(loss)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            
            if info.get("reached", False):
                reached = True
        
        # 软更新 Target Network
        if use_soft_update:
            with torch.no_grad():
                for param, target_param in zip(network.parameters(), target_net.parameters()):
                    target_param.data.copy_(
                        soft_update_tau * param.data + (1 - soft_update_tau) * target_param.data
                    )
        elif ep % target_update_freq == 0:
            target_net.load_state_dict(network.state_dict())
        
        # 学习率调度
        scheduler.step()
        
        # 记录
        logs["episode"].append(ep)
        logs["reward"].append(episode_reward)
        logs["steps"].append(episode_steps)
        logs["epsilon"].append(epsilon)
        logs["loss"].append(np.mean(episode_losses) if episode_losses else 0.0)
        logs["reached"].append(int(reached))
        logs["lr"].append(scheduler.get_last_lr()[0])
        
        # 保存最佳权重
        if ep >= 100:
            recent_reach = np.mean(logs["reached"][-100:])
            if recent_reach > best_reach_rate:
                best_reach_rate = recent_reach
                best_weights = {k: v.cpu().clone() for k, v in network.state_dict().items()}
        
        # 打印进度
        if ep % log_interval == 0:
            recent_rewards = logs["reward"][-log_interval:]
            recent_reached = logs["reached"][-log_interval:]
            print(
                f"  Episode {ep:4d} | "
                f"Reward: {np.mean(recent_rewards):6.2f} | "
                f"Reach: {np.mean(recent_reached)*100:5.1f}% | "
                f"ε: {epsilon:.3f} | "
                f"Best: {best_reach_rate*100:.1f}%"
            )
    
    elapsed = time.time() - start_time
    
    # 恢复最佳权重
    if best_weights is not None:
        network.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
        print(f"  恢复最佳权重 (reach rate: {best_reach_rate*100:.1f}%)")
    
    # 计算最终指标（使用最佳权重评估）
    eval_reached = []
    eval_rewards = []
    for _ in range(100):
        obs = env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(1)
            with torch.no_grad():
                q_values = network(state_tensor)
            action = int(torch.argmax(q_values).item())  # Greedy
            obs, reward, done, info = env.step(action)
            ep_reward += reward
        eval_reached.append(int(info.get("reached", False)))
        eval_rewards.append(ep_reward)
    
    final_reach = np.mean(eval_reached) * 100
    final_reward = np.mean(eval_rewards)
    
    results = {
        "name": network_name,
        "params": sum(p.numel() for p in network.parameters() if p.requires_grad),
        "training_time": elapsed,
        "final_reward": final_reward,
        "final_reach_rate": final_reach,
        "best_reach_rate": best_reach_rate * 100,
        "logs": logs,
    }
    
    print(f"\n  完成! 耗时: {elapsed:.1f}s")
    print(f"  评估到达率 (100 ep): {final_reach:.1f}%")
    print(f"  评估奖励: {final_reward:.2f}")
    
    return results


def plot_comparison_v2(results: List[Dict], save_path: Path):
    """绘制对比图 V2"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    window = 50
    
    def moving_avg(data, w):
        if len(data) < w:
            return data
        cumsum = np.cumsum(np.insert(data, 0, 0))
        return (cumsum[w:] - cumsum[:-w]) / w
    
    # 1. 奖励曲线
    ax = axes[0, 0]
    for i, r in enumerate(results):
        eps = r["logs"]["episode"]
        rewards = r["logs"]["reward"]
        ma = moving_avg(rewards, window)
        ax.plot(eps[window-1:], ma, color=colors[i % len(colors)], 
                label=f'{r["name"]}', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (MA-50)")
    ax.set_title("Training Reward")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 2. 到达率曲线
    ax = axes[0, 1]
    for i, r in enumerate(results):
        eps = r["logs"]["episode"]
        reached = np.array(r["logs"]["reached"]) * 100
        ma = moving_avg(reached, window)
        ax.plot(eps[window-1:], ma, color=colors[i % len(colors)], 
                label=r["name"], linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reach Rate (%)")
    ax.set_title("Target Reach Rate (MA-50)")
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. ε 和学习率曲线
    ax = axes[1, 0]
    for i, r in enumerate(results):
        eps = r["logs"]["episode"]
        epsilon = r["logs"]["epsilon"]
        ax.plot(eps, epsilon, color=colors[i % len(colors)], 
                label=f'{r["name"]} ε', linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (ε)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 4. 最终性能对比
    ax = axes[1, 1]
    names = [r["name"] for r in results]
    final_reach = [r["final_reach_rate"] for r in results]
    best_reach = [r["best_reach_rate"] for r in results]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_reach, width, label='Final (Eval)', color='#2ecc71')
    bars2 = ax.bar(x + width/2, best_reach, width, label='Best (Training)', color='#3498db', alpha=0.7)
    
    ax.set_ylabel('Reach Rate (%)')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=9)
    ax.legend()
    ax.set_ylim([0, 110])
    ax.grid(alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, final_reach):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('DQN Architecture Comparison V2 (with Double DQN + Soft Update)', 
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n保存对比图到: {save_path}")


def print_summary_v2(results: List[Dict]):
    """打印汇总"""
    print("\n" + "=" * 90)
    print("性能对比汇总 (V2 - 改进训练)")
    print("=" * 90)
    print(f"{'网络':<20} {'参数量':>12} {'时间':>8} {'最终到达率':>12} {'最佳到达率':>12} {'最终奖励':>10}")
    print("-" * 90)
    
    for r in results:
        print(
            f"{r['name']:<20} "
            f"{r['params']:>12,} "
            f"{r['training_time']:>7.1f}s "
            f"{r['final_reach_rate']:>11.1f}% "
            f"{r['best_reach_rate']:>11.1f}% "
            f"{r['final_reward']:>10.2f}"
        )
    
    print("=" * 90)
    
    best = max(results, key=lambda x: x['final_reach_rate'])
    print(f"\n🏆 最佳网络: {best['name']} ({best['final_reach_rate']:.1f}%)")


def parse_args():
    p = argparse.ArgumentParser(description="DQN 架构对比 V2")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/architecture_comparison_v2"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if args.quick:
        args.episodes = 1000
        log_interval = 100
    else:
        log_interval = 200
    
    # 环境
    env = RLArm2DEnv(RLArmConfig(max_steps=100, target_radius=0.1))
    state_dim = env.observation_dim
    action_dim = env.action_space_n
    
    print(f"\n环境: state_dim={state_dim}, action_dim={action_dim}")
    print(f"训练 episodes: {args.episodes}")
    print(f"改进: Double DQN, Soft Update, Linear ε decay, Cosine LR")
    
    # 网络
    networks = [
        ("CNN+LSTM", DQNNetwork(state_dim, action_dim, seq_len=1)),
        ("LightTransformer", LightTransformerDQN(state_dim, action_dim, seq_len=1, d_model=64, n_heads=4)),
        ("Transformer", TransformerDQN(state_dim, action_dim, seq_len=1, d_model=64, n_heads=4, n_layers=2)),
    ]
    
    results = []
    for name, network in networks:
        network = network.to(device)
        
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        
        result = train_network_v2(
            network_name=name,
            network=network,
            env=env,
            num_episodes=args.episodes,
            device=device,
            log_interval=log_interval,
        )
        results.append(result)
    
    print_summary_v2(results)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison_v2(results, args.output_dir / "comparison_v2.png")
    
    # 保存结果
    summary = {
        "config": {"episodes": args.episodes, "seed": args.seed, "version": "v2"},
        "improvements": ["Double DQN", "Soft Update", "Linear ε decay", "Cosine LR", "Best weights restore"],
        "results": [
            {k: v for k, v in r.items() if k != "logs"}
            for r in results
        ]
    }
    with open(args.output_dir / "summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n结果保存到: {args.output_dir}")


if __name__ == "__main__":
    main()

