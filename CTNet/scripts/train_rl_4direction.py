#!/usr/bin/env python3
"""
训练 4 方向目标 RL 模型

训练 RL Agent 学习根据目标位置（left/right/up/down）选择正确的动作序列到达目标。

用法：
    python scripts/train_rl_4direction.py --episodes 1000 --device cuda

创建时间: 2026-02-17
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dqn_transformer import TransformerDQN, LightTransformerDQN
from scripts.dqn_model import DQNNetwork, ReplayBuffer


# ============================================================================
# 4 方向目标环境
# ============================================================================

@dataclass
class FourDirectionConfig:
    """4 方向目标环境配置"""
    step_size: float = 0.05
    max_steps: int = 50
    target_radius: float = 0.15
    
    # 4 个方向的目标位置
    targets: Dict[int, Tuple[float, float]] = None
    
    # Reward
    reward_reach: float = 10.0
    reward_step: float = -0.01
    reward_distance_scale: float = 1.0
    reward_wrong_direction: float = -0.1  # 远离目标的惩罚
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = {
                0: (-0.5, 0.0),   # left
                1: (0.5, 0.0),    # right
                2: (0.0, 0.5),    # up
                3: (0.0, -0.5),   # down
            }


class FourDirectionEnv:
    """
    4 方向目标 RL 环境
    
    每个 episode 随机选择一个方向作为目标，
    Agent 需要学习根据目标位置选择正确的动作。
    """
    
    def __init__(self, config: Optional[FourDirectionConfig] = None):
        self.cfg = config or FourDirectionConfig()
        self.action_space_n = 4  # left, right, up, down
        self.observation_dim = 5  # [y, z, target_y, target_z, distance]
        
        self._y = 0.0
        self._z = 0.0
        self._target_y = 0.0
        self._target_z = 0.0
        self._target_direction = 0
        self._step_count = 0
    
    def reset(self, direction: Optional[int] = None) -> np.ndarray:
        """重置环境，随机选择一个方向"""
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
        
        # 随机选择方向
        if direction is None:
            self._target_direction = np.random.randint(4)
        else:
            self._target_direction = direction
        
        self._target_y, self._target_z = self.cfg.targets[self._target_direction]
        
        return self._get_obs()
    
    def _get_obs(self) -> np.ndarray:
        dist = self._distance_to_target()
        return np.array([
            self._y, self._z,
            self._target_y, self._target_z,
            dist
        ], dtype=np.float32)
    
    def _distance_to_target(self) -> float:
        return np.sqrt(
            (self._y - self._target_y) ** 2 +
            (self._z - self._target_z) ** 2
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 0=left(+y), 1=right(-y), 2=up(+z), 3=down(-z)
        """
        prev_dist = self._distance_to_target()
        
        # 动作映射
        dy = self.cfg.step_size if action == 0 else (-self.cfg.step_size if action == 1 else 0.0)
        dz = self.cfg.step_size if action == 2 else (-self.cfg.step_size if action == 3 else 0.0)
        
        # 更新位置
        self._y = np.clip(self._y + dy, -1.0, 1.0)
        self._z = np.clip(self._z + dz, -1.0, 1.0)
        self._step_count += 1
        
        # 计算奖励
        curr_dist = self._distance_to_target()
        reward = self.cfg.reward_step  # 步数惩罚
        
        # 距离改善奖励
        dist_improvement = prev_dist - curr_dist
        reward += dist_improvement * self.cfg.reward_distance_scale
        
        # 如果远离目标，额外惩罚
        if dist_improvement < 0:
            reward += self.cfg.reward_wrong_direction
        
        # 到达目标
        reached = curr_dist < self.cfg.target_radius
        if reached:
            reward += self.cfg.reward_reach
        
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "target_direction": self._target_direction,
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# 训练函数
# ============================================================================

def train_4direction_rl(
    model_type: str = "transformer",
    episodes: int = 1000,
    batch_size: int = 64,
    lr: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay_episodes: int = 800,
    tau: float = 0.005,
    buffer_size: int = 50000,
    device: str = "cuda",
    save_dir: Path = None,
    verbose: bool = True,
):
    """
    训练 4 方向目标 RL 模型
    
    使用 Double DQN + Soft Update + Linear Epsilon Decay
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    if save_dir is None:
        save_dir = _ROOT / "outputs" / "rl_4direction"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 环境
    env = FourDirectionEnv()
    state_dim = env.observation_dim
    action_dim = env.action_space_n
    
    # 模型
    if model_type == "transformer":
        policy_net = TransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
        target_net = TransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    elif model_type == "light_transformer":
        policy_net = LightTransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
        target_net = LightTransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    else:
        policy_net = DQNNetwork(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
        target_net = DQNNetwork(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes, eta_min=lr * 0.1)
    
    replay_buffer = ReplayBuffer(capacity=buffer_size, state_shape=(1, state_dim), device=device)
    
    # 训练记录
    rewards_history = []
    reach_history = []
    loss_history = []
    epsilon_history = []
    direction_reach = {0: [], 1: [], 2: [], 3: []}  # 按方向统计
    
    best_reach_rate = 0.0
    best_weights = None
    
    print("="*60)
    print(f"训练 4 方向目标 RL 模型")
    print("="*60)
    print(f"模型: {model_type}")
    print(f"设备: {device}")
    print(f"Episodes: {episodes}")
    print(f"目标位置:")
    for d, pos in env.cfg.targets.items():
        names = ["left", "right", "up", "down"]
        print(f"  {names[d]}: {pos}")
    print("="*60)
    
    start_time = time.time()
    
    for ep in range(episodes):
        # 线性 epsilon 衰减
        epsilon = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * (ep / epsilon_decay_episodes)
        )
        
        # 每个 episode 随机选择一个方向
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        loss_count = 0
        
        for step in range(env.cfg.max_steps):
            # Epsilon-greedy 动作选择
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                policy_net.eval()
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            # 存入经验回放
            replay_buffer.add(
                state.reshape(1, -1),
                action,
                reward,
                next_state.reshape(1, -1),
                done
            )
            
            episode_reward += reward
            state = next_state
            
            # 训练
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                policy_net.train()
                
                # Double DQN
                with torch.no_grad():
                    # 用 policy net 选择动作
                    next_actions = policy_net(next_states).argmax(dim=1)
                    # 用 target net 评估
                    next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                # 当前 Q 值
                current_q = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                
                # 损失
                loss = F.smooth_l1_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                
                episode_loss += loss.item()
                loss_count += 1
                
                # Soft update target network
                for param, target_param in zip(policy_net.parameters(), target_net.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            if done:
                break
        
        scheduler.step()
        
        # 记录
        rewards_history.append(episode_reward)
        reach_history.append(1 if info["reached"] else 0)
        loss_history.append(episode_loss / max(1, loss_count))
        epsilon_history.append(epsilon)
        direction_reach[info["target_direction"]].append(1 if info["reached"] else 0)
        
        # 检查最佳模型
        if len(reach_history) >= 50:
            recent_reach = np.mean(reach_history[-50:])
            if recent_reach > best_reach_rate:
                best_reach_rate = recent_reach
                best_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
        
        # 打印进度
        if verbose and (ep + 1) % 100 == 0:
            recent_reward = np.mean(rewards_history[-100:])
            recent_reach = np.mean(reach_history[-100:]) * 100
            
            # 按方向统计
            dir_stats = []
            names = ["L", "R", "U", "D"]
            for d in range(4):
                if len(direction_reach[d]) > 0:
                    rate = np.mean(direction_reach[d][-25:]) * 100
                    dir_stats.append(f"{names[d]}:{rate:.0f}%")
            
            print(f"[{ep+1:4d}/{episodes}] reward={recent_reward:6.2f} reach={recent_reach:5.1f}% "
                  f"ε={epsilon:.3f} | {' '.join(dir_stats)}")
    
    elapsed = time.time() - start_time
    
    # 恢复最佳权重
    if best_weights is not None:
        policy_net.load_state_dict(best_weights)
        print(f"\n已恢复最佳权重 (reach rate: {best_reach_rate*100:.1f}%)")
    
    # 最终评估
    print("\n" + "="*60)
    print("最终评估 (100 episodes)")
    print("="*60)
    
    policy_net.eval()
    eval_results = {0: [], 1: [], 2: [], 3: []}
    
    for _ in range(100):
        direction = np.random.randint(4)
        state = env.reset(direction=direction)
        
        for _ in range(env.cfg.max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy_net(state_tensor).argmax(dim=1).item()
            state, _, done, info = env.step(action)
            if done:
                break
        
        eval_results[direction].append(1 if info["reached"] else 0)
    
    names = ["left", "right", "up", "down"]
    total_reached = 0
    for d in range(4):
        rate = np.mean(eval_results[d]) * 100
        total_reached += sum(eval_results[d])
        print(f"  {names[d]:>5}: {rate:5.1f}% ({sum(eval_results[d])}/{len(eval_results[d])})")
    
    total_rate = total_reached / 100 * 100
    print(f"  {'Total':>5}: {total_rate:5.1f}%")
    print(f"\n训练时间: {elapsed:.1f}s")
    print("="*60)
    
    # 保存模型
    model_path = save_dir / f"rl_4direction_{model_type}.pth"
    torch.save(policy_net.state_dict(), model_path)
    print(f"\n模型已保存: {model_path}")
    
    # 保存训练曲线
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 奖励曲线
    ax1 = axes[0, 0]
    window = min(50, len(rewards_history))
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, 'b-', linewidth=1)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 到达率曲线
    ax2 = axes[0, 1]
    smoothed_reach = np.convolve(reach_history, np.ones(window)/window, mode='valid') * 100
    ax2.plot(smoothed_reach, 'g-', linewidth=1)
    ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reach Rate (%)')
    ax2.set_title('Reach Rate', fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    
    # 按方向到达率
    ax3 = axes[1, 0]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']
    for d in range(4):
        if len(direction_reach[d]) > 10:
            smoothed_dir = np.convolve(direction_reach[d], np.ones(10)/10, mode='valid') * 100
            ax3.plot(smoothed_dir, color=colors[d], label=names[d], linewidth=1, alpha=0.8)
    ax3.set_xlabel('Episode (per direction)')
    ax3.set_ylabel('Reach Rate (%)')
    ax3.set_title('Reach Rate by Direction', fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3)
    
    # 最终评估柱状图
    ax4 = axes[1, 1]
    final_rates = [np.mean(eval_results[d]) * 100 for d in range(4)]
    bars = ax4.bar(names, final_rates, color=colors, alpha=0.8)
    ax4.axhline(y=100, color='r', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Reach Rate (%)')
    ax4.set_title('Final Evaluation by Direction', fontweight='bold')
    ax4.set_ylim(0, 110)
    for bar, rate in zip(bars, final_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{rate:.0f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    fig_path = save_dir / "training_curve.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存: {fig_path}")
    
    # 保存训练日志
    log_data = {
        "model_type": model_type,
        "episodes": episodes,
        "final_reach_rate": total_rate,
        "direction_rates": {names[d]: np.mean(eval_results[d]) * 100 for d in range(4)},
        "training_time_seconds": elapsed,
        "best_reach_rate": best_reach_rate * 100,
    }
    with open(save_dir / "training_log.json", "w") as f:
        json.dump(log_data, f, indent=2)
    
    return policy_net, log_data


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="训练 4 方向目标 RL 模型")
    p.add_argument("--model", choices=["dqn", "transformer", "light_transformer"],
                   default="transformer")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    
    train_4direction_rl(
        model_type=args.model,
        episodes=args.episodes,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        verbose=True,
    )


if __name__ == "__main__":
    main()

