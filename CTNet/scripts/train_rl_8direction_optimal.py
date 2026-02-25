#!/usr/bin/env python3
"""
最优方向 RL 训练

核心策略: 强制引导 agent 选择最接近目标方向的动作
通过:
1. 重塑奖励: 强烈惩罚偏离最优方向的动作
2. 动作掩码: 仅允许合理方向的动作参与选择
3. 角度对齐奖励: 动作方向与目标方向夹角越小，奖励越高

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
# 8 方向定义
# ============================================================================

# 动作向量 (单位向量)
ACTION_VECTORS = np.array([
    [-1.0, 0.0],      # 0: left
    [1.0, 0.0],       # 1: right
    [0.0, 1.0],       # 2: up
    [0.0, -1.0],      # 3: down
    [-0.707, 0.707],  # 4: up_left
    [0.707, 0.707],   # 5: up_right
    [-0.707, -0.707], # 6: down_left
    [0.707, -0.707],  # 7: down_right
])

ACTION_NAMES = ["left", "right", "up", "down", 
                "up_left", "up_right", "down_left", "down_right"]


def get_optimal_action(current_y: float, current_z: float, 
                       target_y: float, target_z: float) -> int:
    """计算指向目标的最优动作"""
    dy = target_y - current_y
    dz = target_z - current_z
    
    if abs(dy) < 0.01 and abs(dz) < 0.01:
        return 0  # 已到达
    
    # 目标方向向量
    target_vec = np.array([dy, dz])
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
    
    # 计算与每个动作的点积 (余弦相似度)
    similarities = ACTION_VECTORS @ target_vec
    
    return int(np.argmax(similarities))


def get_action_alignment(action: int, current_y: float, current_z: float,
                         target_y: float, target_z: float) -> float:
    """计算动作与目标方向的对齐度 (0-1)"""
    dy = target_y - current_y
    dz = target_z - current_z
    
    dist = np.sqrt(dy**2 + dz**2)
    if dist < 0.01:
        return 1.0  # 已到达
    
    target_vec = np.array([dy, dz]) / dist
    action_vec = ACTION_VECTORS[action]
    
    # 点积范围 [-1, 1]，转换为 [0, 1]
    alignment = (np.dot(action_vec, target_vec) + 1) / 2
    return float(alignment)


def get_valid_actions_mask(current_y: float, current_z: float,
                           target_y: float, target_z: float,
                           threshold: float = 0.3) -> np.ndarray:
    """
    获取有效动作掩码
    只允许与目标方向夹角小于阈值的动作
    """
    dy = target_y - current_y
    dz = target_z - current_z
    
    dist = np.sqrt(dy**2 + dz**2)
    if dist < 0.01:
        return np.ones(8, dtype=np.float32)
    
    target_vec = np.array([dy, dz]) / dist
    similarities = ACTION_VECTORS @ target_vec
    
    # 只保留相似度 > threshold 的动作 (约 ±70°)
    mask = (similarities > threshold).astype(np.float32)
    
    # 确保至少有一个动作可用
    if mask.sum() == 0:
        mask[np.argmax(similarities)] = 1.0
    
    return mask


# ============================================================================
# 环境
# ============================================================================

@dataclass
class OptimalEnvConfig:
    step_size: float = 0.05
    target_radius: float = 0.15
    max_steps: int = 50
    boundary: float = 1.0
    # 奖励权重
    alignment_reward_scale: float = 0.1   # 对齐奖励
    misalignment_penalty: float = 0.15    # 偏离惩罚
    distance_reward_scale: float = 2.0    # 距离奖励


class OptimalDirectionEnv:
    """
    最优方向引导环境
    """
    
    def __init__(self, cfg: OptimalEnvConfig):
        self.cfg = cfg
        self.observation_dim = 5
        self.action_dim = 8
        
        self._y = 0.0
        self._z = 0.0
        self._target_y = 0.5
        self._target_z = 0.5
        self._step_count = 0
        self._action_history = []
    
    def reset(
        self,
        target_y: Optional[float] = None,
        target_z: Optional[float] = None,
    ) -> np.ndarray:
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
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
    
    def get_valid_actions(self) -> np.ndarray:
        """获取有效动作掩码"""
        return get_valid_actions_mask(
            self._y, self._z, self._target_y, self._target_z
        )
    
    def get_optimal_action(self) -> int:
        """获取最优动作"""
        return get_optimal_action(
            self._y, self._z, self._target_y, self._target_z
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        prev_dist = self._distance_to_target()
        optimal_action = self.get_optimal_action()
        
        # 计算动作与最优方向的对齐度
        alignment = get_action_alignment(
            action, self._y, self._z, self._target_y, self._target_z
        )
        
        # 执行动作
        dy, dz = ACTION_VECTORS[action]
        dy *= self.cfg.step_size
        dz *= self.cfg.step_size
        
        self._y = np.clip(self._y + dy, -self.cfg.boundary, self.cfg.boundary)
        self._z = np.clip(self._z + dz, -self.cfg.boundary, self.cfg.boundary)
        self._step_count += 1
        self._action_history.append(action)
        
        curr_dist = self._distance_to_target()
        
        # === 奖励计算 ===
        
        # 1. 距离奖励
        distance_reward = (prev_dist - curr_dist) * self.cfg.distance_reward_scale
        
        # 2. 对齐奖励/惩罚 (关键!)
        if alignment > 0.8:  # 非常对齐 (约 ±37°)
            alignment_reward = self.cfg.alignment_reward_scale
        elif alignment > 0.5:  # 中等对齐 (约 ±60°)
            alignment_reward = self.cfg.alignment_reward_scale * 0.3
        else:  # 偏离方向
            alignment_reward = -self.cfg.misalignment_penalty * (1 - alignment)
        
        # 3. 步数惩罚
        step_penalty = -0.01
        
        # 4. 到达奖励
        reached = curr_dist < self.cfg.target_radius
        reach_reward = 10.0 if reached else 0.0
        
        # 5. 轨迹直线度奖励 (到达时)
        trajectory_bonus = 0.0
        if reached and len(self._action_history) > 1:
            # 计算理论最短步数
            initial_dist = np.sqrt(self._target_y**2 + self._target_z**2)
            min_steps = int(initial_dist / self.cfg.step_size)
            actual_steps = len(self._action_history)
            
            # 效率奖励: 步数越接近最短，奖励越高
            efficiency = min(1.0, min_steps / max(actual_steps, 1))
            trajectory_bonus = efficiency * 3.0
        
        # 6. 边界惩罚
        boundary_penalty = 0.0
        if abs(self._y) >= self.cfg.boundary * 0.95 or abs(self._z) >= self.cfg.boundary * 0.95:
            boundary_penalty = -0.1
        
        reward = (distance_reward + alignment_reward + step_penalty + 
                  reach_reward + trajectory_bonus + boundary_penalty)
        
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "position": (self._y, self._z),
            "target": (self._target_y, self._target_z),
            "alignment": alignment,
            "optimal_action": optimal_action,
            "action_history": self._action_history.copy(),
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# DQN 网络 (带动作掩码)
# ============================================================================

class MaskedDQN(nn.Module):
    """带动作掩码的 DQN"""
    
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
    
    def forward(self, x: torch.Tensor, 
                action_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        
        q_values = self.net(x)
        
        # 应用动作掩码
        if action_mask is not None:
            # 将无效动作的 Q 值设为很小的负数
            q_values = q_values.masked_fill(action_mask == 0, -1e9)
        
        return q_values


# ============================================================================
# 经验回放
# ============================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(masks, dtype=np.float32),
            np.array(next_masks, dtype=np.float32),
        )
    
    def __len__(self):
        return len(self.buffer)


# ============================================================================
# 训练
# ============================================================================

def train_optimal_dqn(
    episodes: int = 3000,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.02,
    epsilon_decay: float = 0.996,
    target_update_freq: int = 10,
    use_action_mask: bool = True,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[nn.Module, List[float], List[float]]:
    """训练最优方向 DQN"""
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    cfg = OptimalEnvConfig()
    env = OptimalDirectionEnv(cfg)
    
    policy_net = MaskedDQN().to(device)
    target_net = MaskedDQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=1e-5
    )
    
    replay_buffer = ReplayBuffer()
    
    epsilon = epsilon_start
    episode_rewards = []
    episode_alignments = []
    best_score = -float('inf')
    best_weights = None
    
    # 训练目标分布
    target_positions = [
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
        (0.3, 0.3), (-0.3, 0.3), (0.3, -0.3), (-0.3, -0.3),
        (0.6, 0.0), (-0.6, 0.0), (0.0, 0.6), (0.0, -0.6),
        None, None,
    ]
    
    print(f"\n训练最优方向 DQN")
    print(f"动作掩码: {'开启' if use_action_mask else '关闭'}")
    print(f"设备: {device}")
    print(f"Episodes: {episodes}")
    print("-" * 50)
    
    for episode in range(episodes):
        target = random.choice(target_positions)
        if target is None:
            state = env.reset()
        else:
            state = env.reset(target_y=target[0], target_z=target[1])
        
        total_reward = 0.0
        total_alignment = 0.0
        step_count = 0
        done = False
        
        while not done:
            mask = env.get_valid_actions() if use_action_mask else np.ones(8)
            
            # ε-greedy with mask
            if random.random() < epsilon:
                # 随机选择有效动作
                valid_actions = np.where(mask > 0)[0]
                action = np.random.choice(valid_actions)
            else:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_t, mask_t if use_action_mask else None)
                    action = q_values.argmax(dim=1).item()
            
            next_state, reward, done, info = env.step(action)
            next_mask = env.get_valid_actions() if use_action_mask else np.ones(8)
            
            replay_buffer.push(state, action, reward, next_state, done, mask, next_mask)
            
            state = next_state
            total_reward += reward
            total_alignment += info["alignment"]
            step_count += 1
            
            # 训练
            if len(replay_buffer) >= batch_size:
                (states, actions, rewards, next_states, dones, 
                 masks, next_masks) = replay_buffer.sample(batch_size)
                
                states_t = torch.tensor(states, dtype=torch.float32).to(device)
                actions_t = torch.tensor(actions, dtype=torch.long).to(device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32).to(device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32).to(device)
                dones_t = torch.tensor(dones, dtype=torch.float32).to(device)
                masks_t = torch.tensor(masks, dtype=torch.float32).to(device) if use_action_mask else None
                next_masks_t = torch.tensor(next_masks, dtype=torch.float32).to(device) if use_action_mask else None
                
                # 当前 Q 值
                q_values = policy_net(states_t, masks_t)
                q_values = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)
                
                # Double DQN with mask
                with torch.no_grad():
                    next_q_policy = policy_net(next_states_t, next_masks_t)
                    next_actions = next_q_policy.argmax(dim=1)
                    next_q_target = target_net(next_states_t, next_masks_t)
                    next_q_values = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards_t + gamma * next_q_values * (1 - dones_t)
                
                loss = F.smooth_l1_loss(q_values, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
        
        episode_rewards.append(total_reward)
        avg_alignment = total_alignment / max(step_count, 1)
        episode_alignments.append(avg_alignment)
        
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        scheduler.step()
        
        # 记录最佳模型
        if episode >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_align = np.mean(episode_alignments[-100:])
            score = avg_reward + avg_align * 10  # 加权对齐度
            if score > best_score:
                best_score = score
                best_weights = {k: v.cpu().clone() for k, v in policy_net.state_dict().items()}
        
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_align = np.mean(episode_alignments[-100:])
            reach_rate = sum(1 for r in episode_rewards[-100:] if r > 8) / 100
            print(f"Episode {episode+1:4d}: Reward={avg_reward:7.2f}, "
                  f"Align={avg_align:.2f}, Reach={reach_rate:.0%}, ε={epsilon:.3f}")
    
    if best_weights is not None:
        policy_net.load_state_dict(best_weights)
        print(f"\n★ 恢复最佳模型 (score={best_score:.2f})")
    
    return policy_net, episode_rewards, episode_alignments


def evaluate_model(
    model: nn.Module,
    device: torch.device,
    use_action_mask: bool = True,
    n_episodes: int = 50,
) -> Tuple[float, float, List]:
    """评估模型"""
    cfg = OptimalEnvConfig()
    env = OptimalDirectionEnv(cfg)
    
    model.eval()
    reached_count = 0
    alignment_scores = []
    trajectories = []
    
    test_targets = [
        (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5),
        (0.4, 0.4), (-0.4, 0.4), (0.4, -0.4), (-0.4, -0.4),
    ]
    
    for i in range(n_episodes):
        target = test_targets[i % len(test_targets)]
        state = env.reset(target_y=target[0], target_z=target[1])
        
        done = False
        trajectory = [(0.0, 0.0)]
        total_alignment = 0.0
        steps = 0
        
        while not done:
            mask = env.get_valid_actions() if use_action_mask else np.ones(8)
            
            with torch.no_grad():
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mask_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = model(state_t, mask_t if use_action_mask else None)
                action = q_values.argmax(dim=1).item()
            
            state, reward, done, info = env.step(action)
            trajectory.append(info["position"])
            total_alignment += info["alignment"]
            steps += 1
        
        if info["reached"]:
            reached_count += 1
        
        alignment_scores.append(total_alignment / max(steps, 1))
        trajectories.append((target, trajectory))
    
    reach_rate = reached_count / n_episodes
    avg_alignment = np.mean(alignment_scores)
    
    print(f"\n评估结果 ({n_episodes} episodes):")
    print(f"  到达率: {reach_rate:.1%}")
    print(f"  平均对齐度: {avg_alignment:.2f}")
    
    return reach_rate, avg_alignment, trajectories


def visualize_trajectories(trajectories: List, output_path: Path):
    """可视化轨迹"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (target, traj) in enumerate(trajectories[:8]):
        ax = axes[idx]
        
        ys = [p[0] for p in traj]
        zs = [p[1] for p in traj]
        
        # 绘制理想直线
        ax.plot([0, target[0]], [0, target[1]], 'r--', 
                linewidth=2, alpha=0.5, label='Ideal')
        
        # 绘制实际轨迹
        ax.plot(ys, zs, 'b-', alpha=0.8, linewidth=1.5, label='Actual')
        ax.plot(ys[0], zs[0], 'go', markersize=10)
        ax.plot(ys[-1], zs[-1], 'bs', markersize=8)
        ax.plot(target[0], target[1], 'r*', markersize=15)
        
        circle = plt.Circle(target, 0.15, fill=False, color='red', linestyle='--')
        ax.add_patch(circle)
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Target: ({target[0]:.1f}, {target[1]:.1f})')
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle('Optimal Direction DQN - Trajectories vs Ideal Paths', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"轨迹可视化已保存: {output_path}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="最优方向 RL 训练")
    
    p.add_argument("--episodes", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-action-mask", action="store_true",
                   help="禁用动作掩码")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=Path, 
                   default=_ROOT / "outputs" / "rl_8direction_optimal")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    use_action_mask = not args.no_action_mask
    
    model, rewards, alignments = train_optimal_dqn(
        episodes=args.episodes,
        batch_size=args.batch_size,
        lr=args.lr,
        use_action_mask=use_action_mask,
        device=args.device,
    )
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 评估
    reach_rate, avg_align, trajectories = evaluate_model(
        model, device, use_action_mask=use_action_mask, n_episodes=50
    )
    
    # 保存
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "rl_8direction_optimal.pth"
    
    torch.save({
        'state_dict': model.state_dict(),
        'action_dim': 8,
        'use_action_mask': use_action_mask,
        'reach_rate': reach_rate,
        'alignment': avg_align,
    }, model_path)
    
    print(f"\n✅ 模型已保存: {model_path}")
    
    # 可视化
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
    
    smoothed_align = np.convolve(alignments, np.ones(window)/window, mode='valid')
    axes[1].plot(smoothed_align, 'g-', alpha=0.8)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Alignment Score')
    axes[1].set_title('Action Alignment with Optimal Direction')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(args.output_dir / "training_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练曲线已保存: {args.output_dir / 'training_curve.png'}")


if __name__ == "__main__":
    main()
