#!/usr/bin/env python3
"""
8 方向 RL 训练

扩展动作空间为 8 个方向（含对角线）:
    0: left      (-1, 0)
    1: right     (+1, 0)
    2: up        (0, +1)
    3: down      (0, -1)
    4: up_left   (-1, +1) / √2
    5: up_right  (+1, +1) / √2
    6: down_left (-1, -1) / √2
    7: down_right(+1, -1) / √2

创建时间: 2026-02-24
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 项目路径
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ============================================================================
# 8 方向环境
# ============================================================================

@dataclass
class EightDirectionConfig:
    """8 方向环境配置"""
    step_size: float = 0.05
    diagonal_factor: float = 0.707  # 1/√2，保持对角线步长一致
    target_radius: float = 0.15
    max_steps: int = 50
    boundary: float = 1.0


class EightDirectionEnv:
    """
    8 方向 2D 移动环境
    
    动作空间:
        0: left, 1: right, 2: up, 3: down
        4: up_left, 5: up_right, 6: down_left, 7: down_right
    
    状态空间:
        [y, z, target_y, target_z, distance]
    """
    
    # 8 个方向的位移向量
    ACTION_VECTORS = {
        0: (-1.0, 0.0),    # left
        1: (1.0, 0.0),     # right
        2: (0.0, 1.0),     # up
        3: (0.0, -1.0),    # down
        4: (-0.707, 0.707),   # up_left
        5: (0.707, 0.707),    # up_right
        6: (-0.707, -0.707),  # down_left
        7: (0.707, -0.707),   # down_right
    }
    
    ACTION_NAMES = ["left", "right", "up", "down", 
                    "up_left", "up_right", "down_left", "down_right"]
    
    def __init__(self, cfg: EightDirectionConfig):
        self.cfg = cfg
        self.observation_dim = 5
        self.action_dim = 8
        
        self._y = 0.0
        self._z = 0.0
        self._target_y = 0.5
        self._target_z = 0.5
        self._step_count = 0
    
    def reset(
        self,
        target_y: Optional[float] = None,
        target_z: Optional[float] = None,
    ) -> np.ndarray:
        """重置环境"""
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
        
        if target_y is not None:
            self._target_y = target_y
        else:
            self._target_y = random.uniform(-0.6, 0.6)
        
        if target_z is not None:
            self._target_z = target_z
        else:
            self._target_z = random.uniform(-0.6, 0.6)
        
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
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        prev_dist = self._distance_to_target()
        
        # 获取动作向量
        dy, dz = self.ACTION_VECTORS[action]
        dy *= self.cfg.step_size
        dz *= self.cfg.step_size
        
        # 更新位置
        self._y = np.clip(self._y + dy, -self.cfg.boundary, self.cfg.boundary)
        self._z = np.clip(self._z + dz, -self.cfg.boundary, self.cfg.boundary)
        self._step_count += 1
        
        # 计算奖励
        curr_dist = self._distance_to_target()
        
        # 距离奖励
        reward = (prev_dist - curr_dist) * 2.0
        
        # 步数惩罚
        reward -= 0.01
        
        # 到达奖励
        reached = curr_dist < self.cfg.target_radius
        if reached:
            reward += 10.0
        
        # 边界惩罚
        if abs(self._y) >= self.cfg.boundary * 0.95 or abs(self._z) >= self.cfg.boundary * 0.95:
            reward -= 0.1
        
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "position": (self._y, self._z),
            "target": (self._target_y, self._target_z),
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# DQN 网络
# ============================================================================

class DQN8Direction(nn.Module):
    """8 方向 DQN 网络"""
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, state_dim) 或 (batch, state_dim)
        if x.dim() == 3:
            x = x.squeeze(1)
        return self.net(x)


class TransformerDQN8Direction(nn.Module):
    """8 方向 Transformer DQN"""
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(state_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output(x)


# ============================================================================
# 经验回放
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 训练
# ============================================================================

def train_8direction_dqn(
    model_type: str = "dqn",
    episodes: int = 2000,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    训练 8 方向 DQN
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    # 创建环境
    cfg = EightDirectionConfig()
    env = EightDirectionEnv(cfg)
    
    # 创建网络
    if model_type == "transformer":
        policy_net = TransformerDQN8Direction().to(device)
        target_net = TransformerDQN8Direction().to(device)
    else:
        policy_net = DQN8Direction().to(device)
        target_net = DQN8Direction().to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=episodes, eta_min=1e-5)
    
    replay_buffer = ReplayBuffer()
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_reach_rates = []
    best_avg_reward = -float('inf')
    best_weights = None
    
    # 训练目标分布（包含对角线）
    target_positions = [
        # 4 个基本方向
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        # 4 个对角线
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
        # 中间位置
        (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3),
        # 随机位置
        None, None, None, None,
    ]
    
    print(f"\n训练 8 方向 DQN ({model_type})")
    print(f"设备: {device}")
    print(f"Episodes: {episodes}")
    print("-" * 40)
    
    for episode in range(episodes):
        # 选择目标位置
        target = random.choice(target_positions)
        if target is None:
            state = env.reset()
        else:
            state = env.reset(target_y=target[0], target_z=target[1])
        
        total_reward = 0.0
        done = False
        
        while not done:
            # ε-greedy 策略
            if random.random() < epsilon:
                action = random.randint(0, 7)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            next_state, reward, done, info = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            # 训练
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states_t = torch.tensor(states, dtype=torch.float32).to(device)
                actions_t = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).to(device)
                
                # 当前 Q 值
                q_values = policy_net(states_t)
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                
                # Double DQN: 用 policy_net 选动作，用 target_net 评估
                with torch.no_grad():
                    next_actions = policy_net(next_states_t).argmax(dim=1)
                    next_q_values = target_net(next_states_t)
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + gamma * next_q_values * (1 - dones_t)
                
                loss = F.smooth_l1_loss(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        episode_rewards.append(total_reward)
        episode_reach_rates.append(1.0 if info["reached"] else 0.0)
        
        # 更新 epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 更新目标网络
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        scheduler.step()
        
        # 记录最佳模型
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
        
        # 打印进度
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            reach_rate = np.mean(episode_reach_rates[-100:])
            print(f"Episode {episode+1:4d}: Avg Reward={avg_reward:7.2f}, "
                  f"Reach={reach_rate:.1%}, ε={epsilon:.3f}")
    
    # 恢复最佳权重
    if best_weights is not None:
        policy_net.load_state_dict(best_weights)
        print(f"\n★ 恢复最佳模型 (avg_reward={best_avg_reward:.2f})")
    
    return policy_net, episode_rewards, episode_reach_rates


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    n_episodes: int = 100,
    verbose: bool = True,
) -> float:
    """评估模型"""
    cfg = EightDirectionConfig()
    env = EightDirectionEnv(cfg)
    
    model.eval()
    reached_count = 0
    
    # 测试各种目标位置
    test_targets = [
        # 基本方向
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        # 对角线
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
    ]
    
    for i in range(n_episodes):
        target = test_targets[i % len(test_targets)]
        state = env.reset(target_y=target[0], target_z=target[1])
        done = False
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            state, reward, done, info = env.step(action)
        
        if info["reached"]:
            reached_count += 1
    
    reach_rate = reached_count / n_episodes
    
    if verbose:
        print(f"\n评估结果 ({n_episodes} episodes):")
        print(f"  到达率: {reach_rate:.1%}")
    
    return reach_rate


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="8 方向 RL 训练")
    
    p.add_argument("--model", choices=["dqn", "transformer"], default="transformer")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "rl_8direction")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # 训练
    model, rewards, reach_rates = train_8direction_dqn(
        model_type=args.model,
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 评估
    final_reach_rate = evaluate_model(model, device, n_episodes=100)
    
    # 保存模型
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / f"rl_8direction_{args.model}.pth"
    
    torch.save({
        'state_dict': model.state_dict(),
        'model_type': args.model,
        'action_dim': 8,
        'reach_rate': final_reach_rate,
    }, model_path)
    
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 保存训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 奖励曲线
    ax1 = axes[0]
    window = 50
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    ax1.plot(smoothed_rewards, 'b-', alpha=0.8)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Reward (8-Direction)')
    ax1.grid(True, alpha=0.3)
    
    # 到达率曲线
    ax2 = axes[1]
    smoothed_reach = np.convolve(reach_rates, np.ones(window)/window, mode='valid')
    ax2.plot(smoothed_reach, 'g-', alpha=0.8)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reach Rate')
    ax2.set_title('Reach Rate (8-Direction)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    
    curve_path = args.output_dir / f"training_curve_{args.model}.png"
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {curve_path}")


if __name__ == "__main__":
    main()
