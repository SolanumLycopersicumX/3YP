#!/usr/bin/env python3
"""
平滑 8 方向 RL 训练 (Transformer 序列决策)

核心改进:
1. 使用 Transformer 处理历史状态序列，从整体轨迹推断运动方向
2. 增加方向一致性奖励，惩罚频繁转向
3. 使用动作嵌入，让模型学习动作间的关系

创建时间: 2026-02-24
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ============================================================================
# 配置
# ============================================================================

@dataclass
class SmoothEnvConfig:
    step_size: float = 0.05
    target_radius: float = 0.15
    max_steps: int = 50
    boundary: float = 1.0
    # 平滑控制参数
    direction_consistency_reward: float = 0.02  # 方向一致奖励
    direction_change_penalty: float = 0.03      # 方向改变惩罚


# ============================================================================
# 8 方向环境 (带平滑奖励)
# ============================================================================

class SmoothEightDirectionEnv:
    """
    8 方向环境 + 轨迹平滑奖励
    """
    
    ACTION_VECTORS = {
        0: (-1.0, 0.0),      # left
        1: (1.0, 0.0),       # right
        2: (0.0, 1.0),       # up
        3: (0.0, -1.0),      # down
        4: (-0.707, 0.707),  # up_left
        5: (0.707, 0.707),   # up_right
        6: (-0.707, -0.707), # down_left
        7: (0.707, -0.707),  # down_right
    }
    
    ACTION_NAMES = ["left", "right", "up", "down", 
                    "up_left", "up_right", "down_left", "down_right"]
    
    def __init__(self, cfg: SmoothEnvConfig):
        self.cfg = cfg
        self.observation_dim = 5
        self.action_dim = 8
        
        self._y = 0.0
        self._z = 0.0
        self._target_y = 0.5
        self._target_z = 0.5
        self._step_count = 0
        self._prev_action = -1  # 记录上一步动作
        self._action_history = []
    
    def reset(
        self,
        target_y: Optional[float] = None,
        target_z: Optional[float] = None,
    ) -> np.ndarray:
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
        self._prev_action = -1
        self._action_history = []
        
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
    
    def _are_actions_similar(self, a1: int, a2: int) -> bool:
        """判断两个动作是否相似（方向接近）"""
        if a1 == a2:
            return True
        
        # 相似动作组
        similar_groups = [
            {0, 4, 6},  # 左侧方向: left, up_left, down_left
            {1, 5, 7},  # 右侧方向: right, up_right, down_right
            {2, 4, 5},  # 上方方向: up, up_left, up_right
            {3, 6, 7},  # 下方方向: down, down_left, down_right
        ]
        
        for group in similar_groups:
            if a1 in group and a2 in group:
                return True
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_dist = self._distance_to_target()
        
        # 获取动作向量
        dy, dz = self.ACTION_VECTORS[action]
        dy *= self.cfg.step_size
        dz *= self.cfg.step_size
        
        # 更新位置
        self._y = np.clip(self._y + dy, -self.cfg.boundary, self.cfg.boundary)
        self._z = np.clip(self._z + dz, -self.cfg.boundary, self.cfg.boundary)
        self._step_count += 1
        
        # 记录动作历史
        self._action_history.append(action)
        
        # 计算奖励
        curr_dist = self._distance_to_target()
        
        # 1. 距离奖励
        reward = (prev_dist - curr_dist) * 2.0
        
        # 2. 步数惩罚
        reward -= 0.01
        
        # 3. 方向一致性奖励/惩罚
        if self._prev_action >= 0:
            if action == self._prev_action:
                # 完全相同的动作，给予奖励
                reward += self.cfg.direction_consistency_reward
            elif self._are_actions_similar(action, self._prev_action):
                # 相似方向，小奖励
                reward += self.cfg.direction_consistency_reward * 0.5
            else:
                # 方向改变，惩罚
                reward -= self.cfg.direction_change_penalty
        
        # 4. 到达奖励
        reached = curr_dist < self.cfg.target_radius
        if reached:
            reward += 10.0
            # 额外奖励：轨迹平滑度（方向改变次数少）
            if len(self._action_history) > 1:
                changes = sum(1 for i in range(1, len(self._action_history)) 
                             if self._action_history[i] != self._action_history[i-1])
                smoothness_bonus = max(0, 1.0 - changes / len(self._action_history))
                reward += smoothness_bonus * 2.0
        
        # 5. 边界惩罚
        if abs(self._y) >= self.cfg.boundary * 0.95 or abs(self._z) >= self.cfg.boundary * 0.95:
            reward -= 0.1
        
        self._prev_action = action
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "position": (self._y, self._z),
            "target": (self._target_y, self._target_z),
            "action_history": self._action_history.copy(),
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# Transformer 序列 DQN
# ============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SequenceTransformerDQN(nn.Module):
    """
    序列 Transformer DQN
    
    从历史状态序列中学习平滑的运动策略。
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        action_dim: int = 8,
        seq_len: int = 10,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 状态编码
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # 动作嵌入 (用于编码历史动作)
        self.action_embedding = nn.Embedding(action_dim + 1, d_model)  # +1 for no-action
        
        # 状态-动作融合
        self.fusion = nn.Linear(d_model * 2, d_model)
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Q 值输出
        self.q_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )
    
    def forward(
        self, 
        states: torch.Tensor, 
        actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            states: (batch, seq_len, state_dim) 或 (batch, state_dim)
            actions: (batch, seq_len) 历史动作 (可选)
        """
        if states.dim() == 2:
            states = states.unsqueeze(1)
        
        batch_size, seq_len, _ = states.shape
        
        # 编码状态
        state_enc = self.state_encoder(states)  # (batch, seq, d_model)
        
        # 编码动作
        if actions is not None:
            # actions: (batch, seq_len), 值范围 [0, 7] 或 8 表示无动作
            action_enc = self.action_embedding(actions)  # (batch, seq, d_model)
            # 融合状态和动作
            combined = torch.cat([state_enc, action_enc], dim=-1)
            x = self.fusion(combined)
        else:
            # 无动作历史时，用零向量
            action_enc = self.action_embedding(
                torch.full((batch_size, seq_len), self.action_dim, 
                          device=states.device, dtype=torch.long)
            )
            combined = torch.cat([state_enc, action_enc], dim=-1)
            x = self.fusion(combined)
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer
        x = self.transformer(x)
        
        # 使用最后一个位置的输出
        x = x[:, -1, :]
        
        # Q 值
        q_values = self.q_head(x)
        
        return q_values


# ============================================================================
# 序列经验回放
# ============================================================================

class SequenceReplayBuffer:
    """存储状态序列的经验回放"""
    
    def __init__(self, capacity: int = 50000, seq_len: int = 10):
        self.capacity = capacity
        self.seq_len = seq_len
        self.buffer = deque(maxlen=capacity)
    
    def push(
        self, 
        state_seq: np.ndarray,      # (seq_len, state_dim)
        action_seq: np.ndarray,     # (seq_len,)
        action: int,
        reward: float,
        next_state_seq: np.ndarray, # (seq_len, state_dim)
        next_action_seq: np.ndarray,# (seq_len,)
        done: bool
    ):
        self.buffer.append((
            state_seq, action_seq, action, reward, 
            next_state_seq, next_action_seq, done
        ))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state_seqs, action_seqs, actions, rewards, next_state_seqs, next_action_seqs, dones = zip(*batch)
        
        return (
            np.array(state_seqs),
            np.array(action_seqs),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_state_seqs),
            np.array(next_action_seqs),
            np.array(dones, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 训练
# ============================================================================

def train_smooth_transformer(
    seq_len: int = 10,
    episodes: int = 3000,
    batch_size: int = 64,
    lr: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.997,
    target_update_freq: int = 10,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    训练平滑 Transformer DQN
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    cfg = SmoothEnvConfig()
    env = SmoothEightDirectionEnv(cfg)
    
    # 创建网络
    policy_net = SequenceTransformerDQN(
        state_dim=5, action_dim=8, seq_len=seq_len
    ).to(device)
    target_net = SequenceTransformerDQN(
        state_dim=5, action_dim=8, seq_len=seq_len
    ).to(device)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=1e-5
    )
    
    replay_buffer = SequenceReplayBuffer(capacity=100000, seq_len=seq_len)
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_reach_rates = []
    smoothness_scores = []
    best_score = -float('inf')
    best_weights = None
    
    # 训练目标分布
    target_positions = [
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
        (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3),
        None, None,
    ]
    
    print(f"\n训练平滑 Transformer DQN")
    print(f"序列长度: {seq_len}")
    print(f"设备: {device}")
    print(f"Episodes: {episodes}")
    print("-" * 50)
    
    for episode in range(episodes):
        target = random.choice(target_positions)
        if target is None:
            state = env.reset()
        else:
            state = env.reset(target_y=target[0], target_z=target[1])
        
        # 初始化序列缓冲区
        state_buffer = deque(maxlen=seq_len)
        action_buffer = deque(maxlen=seq_len)
        
        # 填充初始序列
        for _ in range(seq_len):
            state_buffer.append(state)
            action_buffer.append(8)  # 8 = no action
        
        total_reward = 0.0
        done = False
        
        while not done:
            state_seq = np.array(state_buffer)
            action_seq = np.array(action_buffer)
            
            # ε-greedy
            if random.random() < epsilon:
                action = random.randint(0, 7)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
                    action_t = torch.tensor(action_seq, dtype=torch.long).unsqueeze(0).to(device)
                    q_values = policy_net(state_t, action_t)
                    action = q_values.argmax(dim=1).item()
            
            next_state, reward, done, info = env.step(action)
            
            # 更新序列
            next_state_buffer = state_buffer.copy()
            next_action_buffer = action_buffer.copy()
            next_state_buffer.append(next_state)
            next_action_buffer.append(action)
            
            next_state_seq = np.array(next_state_buffer)
            next_action_seq = np.array(next_action_buffer)
            
            # 存储经验
            replay_buffer.push(
                state_seq, action_seq, action, reward,
                next_state_seq, next_action_seq, done
            )
            
            # 更新缓冲区
            state_buffer.append(next_state)
            action_buffer.append(action)
            state = next_state
            total_reward += reward
            
            # 训练
            if len(replay_buffer) >= batch_size:
                (state_seqs, action_seqs, actions, rewards, 
                 next_state_seqs, next_action_seqs, dones) = replay_buffer.sample(batch_size)
                
                state_seqs_t = torch.tensor(state_seqs, dtype=torch.float32).to(device)
                action_seqs_t = torch.tensor(action_seqs, dtype=torch.long).to(device)
                actions_t = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_state_seqs_t = torch.tensor(next_state_seqs, dtype=torch.float32).to(device)
                next_action_seqs_t = torch.tensor(next_action_seqs, dtype=torch.long).to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).to(device)
                
                # 当前 Q 值
                q_values = policy_net(state_seqs_t, action_seqs_t)
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                
                # Double DQN
                with torch.no_grad():
                    next_actions = policy_net(next_state_seqs_t, next_action_seqs_t).argmax(dim=1)
                    next_q_values = target_net(next_state_seqs_t, next_action_seqs_t)
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + gamma * next_q_values * (1 - dones_t)
                
                loss = F.smooth_l1_loss(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        episode_rewards.append(total_reward)
        episode_reach_rates.append(1.0 if info["reached"] else 0.0)
        
        # 计算平滑度
        action_hist = info.get("action_history", [])
        if len(action_hist) > 1:
            changes = sum(1 for i in range(1, len(action_hist)) 
                         if action_hist[i] != action_hist[i-1])
            smoothness = 1.0 - changes / len(action_hist)
        else:
            smoothness = 1.0
        smoothness_scores.append(smoothness)
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        scheduler.step()
        
        # 记录最佳模型 (综合评分：奖励 + 平滑度)
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_smoothness = np.mean(smoothness_scores[-100:])
            score = avg_reward + avg_smoothness * 5  # 加权平滑度
            if score > best_score:
                best_score = score
                best_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            reach_rate = np.mean(episode_reach_rates[-100:])
            avg_smooth = np.mean(smoothness_scores[-100:])
            print(f"Episode {episode+1:4d}: Reward={avg_reward:7.2f}, "
                  f"Reach={reach_rate:.0%}, Smooth={avg_smooth:.2f}, ε={epsilon:.3f}")
    
    if best_weights is not None:
        policy_net.load_state_dict(best_weights)
        print(f"\n★ 恢复最佳模型 (score={best_score:.2f})")
    
    return policy_net, episode_rewards, smoothness_scores


def evaluate_smoothness(
    model: nn.Module,
    device: torch.device,
    seq_len: int = 10,
    n_episodes: int = 50,
) -> Tuple[float, float, List]:
    """评估模型的到达率和平滑度"""
    cfg = SmoothEnvConfig()
    env = SmoothEightDirectionEnv(cfg)
    
    model.eval()
    reached_count = 0
    smoothness_scores = []
    trajectories = []
    
    test_targets = [
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
    ]
    
    for i in range(n_episodes):
        target = test_targets[i % len(test_targets)]
        state = env.reset(target_y=target[0], target_z=target[1])
        
        state_buffer = deque(maxlen=seq_len)
        action_buffer = deque(maxlen=seq_len)
        for _ in range(seq_len):
            state_buffer.append(state)
            action_buffer.append(8)
        
        done = False
        trajectory = [(0.0, 0.0)]
        
        while not done:
            with torch.no_grad():
                state_seq = np.array(state_buffer)
                action_seq = np.array(action_buffer)
                state_t = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
                action_t = torch.tensor(action_seq, dtype=torch.long).unsqueeze(0).to(device)
                q_values = model(state_t, action_t)
                action = q_values.argmax(dim=1).item()
            
            state, reward, done, info = env.step(action)
            state_buffer.append(state)
            action_buffer.append(action)
            trajectory.append(info["position"])
        
        if info["reached"]:
            reached_count += 1
        
        action_hist = info.get("action_history", [])
        if len(action_hist) > 1:
            changes = sum(1 for j in range(1, len(action_hist)) 
                         if action_hist[j] != action_hist[j-1])
            smoothness = 1.0 - changes / len(action_hist)
        else:
            smoothness = 1.0
        smoothness_scores.append(smoothness)
        trajectories.append((target, trajectory))
    
    reach_rate = reached_count / n_episodes
    avg_smoothness = np.mean(smoothness_scores)
    
    print(f"\n评估结果 ({n_episodes} episodes):")
    print(f"  到达率: {reach_rate:.1%}")
    print(f"  平均平滑度: {avg_smoothness:.2f}")
    
    return reach_rate, avg_smoothness, trajectories


def visualize_trajectories(trajectories: List, output_path: Path):
    """可视化轨迹对比"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (target, traj) in enumerate(trajectories[:8]):
        ax = axes[idx]
        
        ys = [p[0] for p in traj]
        zs = [p[1] for p in traj]
        
        ax.plot(ys, zs, 'b-', alpha=0.7, linewidth=1.5)
        ax.plot(ys[0], zs[0], 'go', markersize=10, label='Start')
        ax.plot(ys[-1], zs[-1], 'bs', markersize=8, label='End')
        ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')
        
        circle = plt.Circle(target, 0.15, fill=False, color='red', linestyle='--')
        ax.add_patch(circle)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Target: ({target[0]:.1f}, {target[1]:.1f})')
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Smooth Transformer DQN - Trajectory Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"轨迹可视化已保存: {output_path}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="平滑 8 方向 RL 训练")
    
    p.add_argument("--seq-len", type=int, default=10, help="历史序列长度")
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "rl_8direction_smooth")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    model, rewards, smoothness = train_smooth_transformer(
        seq_len=args.seq_len,
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 评估
    reach_rate, avg_smooth, trajectories = evaluate_smoothness(
        model, device, seq_len=args.seq_len, n_episodes=50
    )
    
    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "rl_8direction_smooth_transformer.pth"
    
    torch.save({
        'state_dict': model.state_dict(),
        'seq_len': args.seq_len,
        'action_dim': 8,
        'reach_rate': reach_rate,
        'smoothness': avg_smooth,
    }, model_path)
    
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 可视化轨迹
    visualize_trajectories(trajectories, args.output_dir / "trajectory_examples.png")
    
    # 训练曲线
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    window = 50
    smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
    axes[0].plot(smoothed_rewards, 'b-', alpha=0.8)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('Training Reward')
    axes[0].grid(True, alpha=0.3)
    
    smoothed_smooth = np.convolve(smoothness, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_smooth, 'g-', alpha=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Smoothness Score')
    axes[1].set_title('Trajectory Smoothness')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {args.output_dir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
