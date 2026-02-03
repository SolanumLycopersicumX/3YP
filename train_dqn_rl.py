#!/usr/bin/env python3
"""
真正的 DQN 强化学习训练脚本

与监督学习版本（train_dqn_quick.py）不同，本脚本实现：
1. 环境交互 - Agent 在环境中执行动作，收集经验
2. 经验回放 - 存储 (s, a, r, s', done) 元组
3. Target Network - 使用独立的目标网络稳定训练
4. ε-greedy 探索 - 从高探索逐步衰减到低探索
5. Reward Shaping - 针对机械臂控制的奖励设计

示例：
  python scripts/train_dqn_rl.py \
    --env simple2d --episodes 1000 --device cuda \
    --save-path outputs/dqn_rl_policy.pth

教授反馈整合：
  - Task 1: CNN→RL 替换 ✓ 本脚本核心目标
  - Task 3: Smoother - 增加动作重复惩罚
  - Task 6: 扩展动作空间 - 通过 --n-actions 参数支持
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dqn_model import DQNNetwork, ReplayBuffer, epsilon_greedy_action, dqn_training_step


# ============================================================================
# 增强版环境：带目标点和 Reward Shaping
# ============================================================================
@dataclass
class RLArmConfig:
    """RL 训练专用环境配置"""
    step_size: float = 0.05
    max_steps: int = 100
    y_limits: Tuple[float, float] = (-1.0, 1.0)
    z_limits: Tuple[float, float] = (-1.0, 1.0)
    
    # 目标相关
    target_radius: float = 0.1  # 到达目标的判定半径
    random_target: bool = True  # 是否随机生成目标
    fixed_target: Tuple[float, float] = (0.5, 0.5)  # 固定目标位置
    
    # Reward 设计
    reward_reach: float = 10.0      # 到达目标奖励
    reward_step: float = -0.01      # 每步惩罚（鼓励快速完成）
    reward_distance_scale: float = 1.0  # 距离变化奖励系数
    reward_oscillation: float = -0.5    # 振荡惩罚（左右左类动作）
    reward_boundary: float = -0.5       # 超出边界惩罚
    
    # 关节限位（教授 Task 2）
    enable_limits: bool = True
    soft_limit_margin: float = 0.1  # 软限位边界


class RLArm2DEnv:
    """带 Reward Shaping 的 2D 机械臂 RL 环境"""
    
    def __init__(self, config: Optional[RLArmConfig] = None):
        self.cfg = config or RLArmConfig()
        self.action_space_n = 4  # left, right, up, down
        
        # 状态: [y, z, target_y, target_z, dist_to_target]
        self.observation_dim = 5
        
        self._reset_state()
    
    def _reset_state(self):
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
        self._last_action = None
        self._action_history = []
        
        # 生成目标
        if self.cfg.random_target:
            margin = 0.2
            self._target_y = np.random.uniform(
                self.cfg.y_limits[0] + margin, 
                self.cfg.y_limits[1] - margin
            )
            self._target_z = np.random.uniform(
                self.cfg.z_limits[0] + margin,
                self.cfg.z_limits[1] - margin
            )
        else:
            self._target_y, self._target_z = self.cfg.fixed_target
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            np.random.seed(seed)
        self._reset_state()
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
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        prev_dist = self._distance_to_target()
        
        # 动作映射: 0=left(+y), 1=right(-y), 2=up(+z), 3=down(-z)
        dy = self.cfg.step_size if action == 0 else (-self.cfg.step_size if action == 1 else 0.0)
        dz = self.cfg.step_size if action == 2 else (-self.cfg.step_size if action == 3 else 0.0)
        
        # 更新位置（带限位）
        new_y = self._y + dy
        new_z = self._z + dz
        
        # 边界检查
        hit_boundary = False
        if new_y < self.cfg.y_limits[0] or new_y > self.cfg.y_limits[1]:
            hit_boundary = True
            new_y = np.clip(new_y, *self.cfg.y_limits)
        if new_z < self.cfg.z_limits[0] or new_z > self.cfg.z_limits[1]:
            hit_boundary = True
            new_z = np.clip(new_z, *self.cfg.z_limits)
        
        self._y = new_y
        self._z = new_z
        self._step_count += 1
        
        # 记录动作历史（用于检测振荡）
        self._action_history.append(action)
        if len(self._action_history) > 5:
            self._action_history.pop(0)
        
        # 计算 Reward
        reward = 0.0
        
        # 1. 每步惩罚
        reward += self.cfg.reward_step
        
        # 2. 距离变化奖励（靠近目标为正）
        curr_dist = self._distance_to_target()
        dist_improvement = prev_dist - curr_dist
        reward += dist_improvement * self.cfg.reward_distance_scale
        
        # 3. 到达目标大奖励
        reached = curr_dist < self.cfg.target_radius
        if reached:
            reward += self.cfg.reward_reach
        
        # 4. 边界惩罚
        if hit_boundary:
            reward += self.cfg.reward_boundary
        
        # 5. 振荡惩罚（检测 A-B-A 模式，如左-右-左）
        if self._detect_oscillation():
            reward += self.cfg.reward_oscillation
        
        # 终止条件
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "hit_boundary": hit_boundary,
        }
        
        self._last_action = action
        
        return self._get_obs(), reward, done, info
    
    def _detect_oscillation(self) -> bool:
        """检测 A-B-A 类振荡模式（如 左-右-左）"""
        if len(self._action_history) < 3:
            return False
        
        # 检查最近3步是否形成 A-B-A
        a1, a2, a3 = self._action_history[-3:]
        
        # 左右振荡: 0-1-0 或 1-0-1
        if (a1 == 0 and a2 == 1 and a3 == 0) or (a1 == 1 and a2 == 0 and a3 == 1):
            return True
        # 上下振荡: 2-3-2 或 3-2-3
        if (a1 == 2 and a2 == 3 and a3 == 2) or (a1 == 3 and a2 == 2 and a3 == 3):
            return True
        
        return False


# ============================================================================
# EEG 环境包装器：将 EEG 分类映射到动作
# ============================================================================
class EEGActionEnv:
    """
    将 EEG 分类结果作为"建议动作"的环境包装器。
    
    RL Agent 学习：
    - 何时跟随 EEG 分类结果
    - 何时修正/平滑动作
    - 如何处理不可靠的分类
    
    这实现了 CNN→RL 的替换（教授 Task 1）
    """
    
    def __init__(
        self, 
        base_env: RLArm2DEnv,
        eeg_predictions: np.ndarray,  # 预计算的 EEG 分类结果
        eeg_confidences: Optional[np.ndarray] = None,  # 分类置信度
    ):
        self.base_env = base_env
        self.eeg_predictions = eeg_predictions
        self.eeg_confidences = eeg_confidences
        self._eeg_idx = 0
        
        # 观测维度 = base_obs + eeg_suggestion + confidence
        self.observation_dim = base_env.observation_dim + 2
        self.action_space_n = base_env.action_space_n
    
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        self._eeg_idx = 0
        base_obs = self.base_env.reset(seed)
        return self._augment_obs(base_obs)
    
    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        # 添加 EEG 建议动作和置信度到观测
        if self._eeg_idx < len(self.eeg_predictions):
            eeg_action = float(self.eeg_predictions[self._eeg_idx])
            confidence = (
                float(self.eeg_confidences[self._eeg_idx])
                if self.eeg_confidences is not None
                else 1.0
            )
        else:
            eeg_action = 0.0
            confidence = 0.0
        
        return np.concatenate([base_obs, [eeg_action, confidence]]).astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self.base_env.step(action)
        self._eeg_idx += 1
        
        # 添加 EEG 建议到 info
        if self._eeg_idx < len(self.eeg_predictions):
            info["eeg_suggestion"] = int(self.eeg_predictions[self._eeg_idx])
        
        return self._augment_obs(obs), reward, done, info


# ============================================================================
# 训练逻辑
# ============================================================================
def train_dqn(
    env,
    policy_net: DQNNetwork,
    target_net: DQNNetwork,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    *,
    num_episodes: int = 1000,
    batch_size: int = 64,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 10,
    min_buffer_size: int = 1000,
    device: torch.device = torch.device("cpu"),
    log_interval: int = 50,
    save_path: Optional[Path] = None,
    save_interval: int = 100,
) -> Dict[str, list]:
    """
    完整的 DQN 训练循环
    
    Returns:
        训练日志字典
    """
    logs = {
        "episode": [],
        "reward": [],
        "steps": [],
        "epsilon": [],
        "loss": [],
        "reached": [],
    }
    
    epsilon = epsilon_start
    total_steps = 0
    
    for ep in range(1, num_episodes + 1):
        obs = env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_losses = []
        reached = False
        
        done = False
        while not done:
            # 将观测转换为网络输入格式
            state_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
            # DQN 期望 (batch, seq_len, state_dim)，这里 seq_len=1
            state_tensor = state_tensor.unsqueeze(0).unsqueeze(1)
            
            # ε-greedy 动作选择
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = epsilon_greedy_action(q_values, epsilon)
            
            # 执行动作
            next_obs, reward, done, info = env.step(action)
            
            # 存储经验
            state_for_buffer = obs.reshape(1, -1)  # (1, state_dim)
            next_state_for_buffer = next_obs.reshape(1, -1)
            replay_buffer.add(state_for_buffer, action, reward, next_state_for_buffer, float(done))
            
            # 训练（如果 buffer 足够大）
            if replay_buffer.size >= min_buffer_size:
                batch = replay_buffer.sample(batch_size)
                loss = dqn_training_step(policy_net, target_net, optimizer, batch, gamma=gamma)
                episode_losses.append(loss)
            
            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            if info.get("reached", False):
                reached = True
        
        # 更新 epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # 更新 target network
        if ep % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # 记录日志
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        logs["episode"].append(ep)
        logs["reward"].append(episode_reward)
        logs["steps"].append(episode_steps)
        logs["epsilon"].append(epsilon)
        logs["loss"].append(avg_loss)
        logs["reached"].append(int(reached))
        
        # 打印进度
        if ep % log_interval == 0:
            recent_rewards = logs["reward"][-log_interval:]
            recent_reached = logs["reached"][-log_interval:]
            print(
                f"Episode {ep}/{num_episodes} | "
                f"Avg Reward: {np.mean(recent_rewards):.2f} | "
                f"Reach Rate: {np.mean(recent_reached)*100:.1f}% | "
                f"ε: {epsilon:.3f} | "
                f"Loss: {avg_loss:.4f}"
            )
        
        # 保存 checkpoint
        if save_path and ep % save_interval == 0:
            checkpoint = {
                "episode": ep,
                "policy_state_dict": policy_net.state_dict(),
                "target_state_dict": target_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epsilon": epsilon,
                "logs": logs,
            }
            torch.save(checkpoint, save_path)
    
    return logs


def parse_args():
    p = argparse.ArgumentParser(description="DQN 强化学习训练（真正的 RL）")
    
    # 环境设置
    p.add_argument("--env", choices=["simple2d", "eeg"], default="simple2d",
                   help="simple2d: 纯 RL 环境; eeg: 带 EEG 建议的环境")
    p.add_argument("--max-steps", type=int, default=100, help="每个 episode 最大步数")
    p.add_argument("--target-radius", type=float, default=0.1, help="到达目标判定半径")
    
    # 训练参数
    p.add_argument("--episodes", type=int, default=1000, help="训练 episode 数")
    p.add_argument("--batch-size", type=int, default=64, help="训练 batch 大小")
    p.add_argument("--buffer-size", type=int, default=100000, help="Replay buffer 容量")
    p.add_argument("--min-buffer", type=int, default=1000, help="开始训练前最小 buffer 大小")
    p.add_argument("--lr", type=float, default=1e-3, help="学习率")
    p.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    
    # Epsilon 探索
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    
    # 网络结构
    p.add_argument("--conv-channels", type=int, default=64)
    p.add_argument("--lstm-hidden", type=int, default=128)
    p.add_argument("--fc-hidden", type=int, default=128)
    p.add_argument("--n-actions", type=int, default=4, help="动作空间大小（扩展时修改）")
    
    # 保存/加载
    p.add_argument("--save-path", type=Path, default=Path("outputs/dqn_rl_policy.pth"))
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--resume", type=Path, default=None, help="从 checkpoint 恢复训练")
    
    # 其他
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=50)
    p.add_argument("--target-update", type=int, default=10, help="Target network 更新频率")
    
    # EEG 相关（仅当 --env eeg 时使用）
    p.add_argument("--eeg-preds", type=Path, default=None, help="预计算的 EEG 分类结果 .npy 文件")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建环境
    env_config = RLArmConfig(
        max_steps=args.max_steps,
        target_radius=args.target_radius,
    )
    base_env = RLArm2DEnv(env_config)
    
    if args.env == "eeg" and args.eeg_preds is not None:
        # 加载预计算的 EEG 分类结果
        eeg_preds = np.load(args.eeg_preds)
        env = EEGActionEnv(base_env, eeg_preds)
        state_dim = env.observation_dim
    else:
        env = base_env
        state_dim = base_env.observation_dim
    
    n_actions = args.n_actions
    
    print(f"Environment: {args.env}")
    print(f"State dim: {state_dim}, Action dim: {n_actions}")
    
    # 创建网络
    # 注意: DQNNetwork 期望 (batch, seq_len, state_dim)
    policy_net = DQNNetwork(
        state_dim=state_dim,
        action_dim=n_actions,
        seq_len=1,  # 单步输入
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        fc_hidden=args.fc_hidden,
    ).to(device)
    
    target_net = DQNNetwork(
        state_dim=state_dim,
        action_dim=n_actions,
        seq_len=1,
        conv_channels=args.conv_channels,
        lstm_hidden=args.lstm_hidden,
        fc_hidden=args.fc_hidden,
    ).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # 创建优化器和 Replay Buffer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=args.lr)
    
    replay_buffer = ReplayBuffer(
        capacity=args.buffer_size,
        state_shape=(1, state_dim),  # (seq_len, state_dim)
        device=device,
    )
    
    # 恢复 checkpoint（如果有）
    start_epsilon = args.epsilon_start
    if args.resume and args.resume.exists():
        checkpoint = torch.load(args.resume, map_location=device)
        policy_net.load_state_dict(checkpoint["policy_state_dict"])
        target_net.load_state_dict(checkpoint["target_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epsilon = checkpoint.get("epsilon", args.epsilon_start)
        print(f"Resumed from {args.resume}, episode {checkpoint.get('episode', 0)}")
    
    # 创建保存目录
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 训练
    print(f"\n{'='*60}")
    print("开始 DQN 强化学习训练")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    logs = train_dqn(
        env=env,
        policy_net=policy_net,
        target_net=target_net,
        replay_buffer=replay_buffer,
        optimizer=optimizer,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=start_epsilon,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_freq=args.target_update,
        min_buffer_size=args.min_buffer,
        device=device,
        log_interval=args.log_interval,
        save_path=args.save_path,
        save_interval=args.save_interval,
    )
    
    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed/60:.1f} 分钟")
    
    # 保存最终模型和日志
    final_checkpoint = {
        "episode": args.episodes,
        "policy_state_dict": policy_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epsilon": args.epsilon_end,
        "logs": logs,
        "config": vars(args),
    }
    torch.save(final_checkpoint, args.save_path)
    print(f"保存模型到: {args.save_path}")
    
    # 保存训练曲线
    log_path = args.save_path.with_suffix(".json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in logs.items()}, f, indent=2)
    print(f"保存日志到: {log_path}")
    
    # 打印最终统计
    print(f"\n{'='*60}")
    print("训练统计")
    print(f"{'='*60}")
    print(f"平均奖励 (最后100 ep): {np.mean(logs['reward'][-100:]):.2f}")
    print(f"到达率 (最后100 ep): {np.mean(logs['reached'][-100:])*100:.1f}%")
    print(f"平均步数 (最后100 ep): {np.mean(logs['steps'][-100:]):.1f}")


if __name__ == "__main__":
    main()

