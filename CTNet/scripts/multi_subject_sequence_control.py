#!/usr/bin/env python3
"""
Multi-Subject Sequence Control

为不同被试设计不同的机械臂动作序列，并进行物理控制测试。
每个被试有独特的动作序列，系统使用RL控制机械臂完成序列。

动作序列示例:
    Subject 1: center → right → left → center
    Subject 2: center → left → right → center
    Subject 3: center → up → down → center
    Subject 4: center → down → up → center
    Subject 5: center → right → up → left → down → center (复杂路径)

创建时间: 2026-02-24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt

# 项目路径
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# 动作序列定义
# ============================================================================

@dataclass
class MovementTarget:
    """单个目标位置"""
    name: str           # 位置名称 (e.g., "right", "left", "up", "down", "center")
    y: float            # Y 坐标 (-1 to 1, 负=left, 正=right)
    z: float            # Z 坐标 (-1 to 1, 负=down, 正=up)
    
    @staticmethod
    def from_name(name: str) -> "MovementTarget":
        """从名称创建目标位置"""
        positions = {
            "center": (0.0, 0.0),
            "left":   (-0.5, 0.0),
            "right":  (0.5, 0.0),
            "up":     (0.0, 0.5),
            "down":   (0.0, -0.5),
            # 对角线位置
            "up_left":    (-0.4, 0.4),
            "up_right":   (0.4, 0.4),
            "down_left":  (-0.4, -0.4),
            "down_right": (0.4, -0.4),
        }
        if name not in positions:
            raise ValueError(f"Unknown position: {name}")
        y, z = positions[name]
        return MovementTarget(name=name, y=y, z=z)


@dataclass
class SubjectSequence:
    """被试的动作序列"""
    subject_id: int
    sequence: List[MovementTarget]
    description: str = ""
    
    @property
    def sequence_names(self) -> List[str]:
        return [t.name for t in self.sequence]


# 预定义的被试动作序列
SUBJECT_SEQUENCES = {
    1: SubjectSequence(
        subject_id=1,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("center"),
        ],
        description="Horizontal sweep: center → right → left → center"
    ),
    2: SubjectSequence(
        subject_id=2,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("center"),
        ],
        description="Reverse horizontal: center → left → right → center"
    ),
    3: SubjectSequence(
        subject_id=3,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("center"),
        ],
        description="Vertical sweep: center → up → down → center"
    ),
    4: SubjectSequence(
        subject_id=4,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("center"),
        ],
        description="Reverse vertical: center → down → up → center"
    ),
    5: SubjectSequence(
        subject_id=5,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("center"),
        ],
        description="Square path (clockwise): center → right → up → left → down → center"
    ),
    6: SubjectSequence(
        subject_id=6,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up_right"),
            MovementTarget.from_name("down_left"),
            MovementTarget.from_name("center"),
        ],
        description="Diagonal sweep 1: center → up_right → down_left → center"
    ),
    7: SubjectSequence(
        subject_id=7,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up_left"),
            MovementTarget.from_name("down_right"),
            MovementTarget.from_name("center"),
        ],
        description="Diagonal sweep 2: center → up_left → down_right → center"
    ),
    8: SubjectSequence(
        subject_id=8,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("center"),
        ],
        description="Square path (counter-clockwise): center → left → up → right → down → center"
    ),
}


# ============================================================================
# RL 控制环境 (简化版)
# ============================================================================

class SequenceControlEnv:
    """
    序列控制环境
    
    控制机械臂依次到达一系列目标位置。
    支持 4 方向 或 8 方向动作空间。
    """
    
    # 8 方向动作向量
    ACTION_VECTORS_8 = {
        0: (-1.0, 0.0),      # left
        1: (1.0, 0.0),       # right
        2: (0.0, 1.0),       # up
        3: (0.0, -1.0),      # down
        4: (-0.707, 0.707),  # up_left
        5: (0.707, 0.707),   # up_right
        6: (-0.707, -0.707), # down_left
        7: (0.707, -0.707),  # down_right
    }
    
    def __init__(
        self,
        serial_env,
        target_radius: float = 0.15,
        step_size: float = 0.05,
        max_step_size: float = 0.15,
        adaptive_step: bool = False,
        max_steps_per_target: int = 30,
        use_8_direction: bool = False,
    ):
        self.serial_env = serial_env
        self.target_radius = target_radius
        self.step_size = step_size
        self.max_step_size = max_step_size
        self.adaptive_step = adaptive_step
        self.max_steps_per_target = max_steps_per_target
        self.use_8_direction = use_8_direction
        self.action_dim = 8 if use_8_direction else 4
        
        # 状态
        self._y = 0.0
        self._z = 0.0
        self._target_y = 0.0
        self._target_z = 0.0
        self._step_count = 0
        
        # 记录
        self.trajectory = []
        self.target_history = []
        
    def reset(self, start_y: float = 0.0, start_z: float = 0.0):
        """重置环境"""
        self._y = start_y
        self._z = start_z
        self._step_count = 0
        self.trajectory = [(start_y, start_z)]
        self.target_history = []
        
        if self.serial_env is not None:
            self.serial_env.reset()
        
        return self._get_obs()
    
    def set_target(self, target_y: float, target_z: float):
        """设置新的目标位置"""
        self._target_y = target_y
        self._target_z = target_z
        self._step_count = 0
        self.target_history.append((target_y, target_z))
    
    def _get_obs(self) -> np.ndarray:
        """获取观测"""
        dist = self._distance_to_target()
        return np.array([
            self._y, self._z,
            self._target_y, self._target_z,
            dist
        ], dtype=np.float32)
    
    def _distance_to_target(self) -> float:
        """计算到目标的距离"""
        return np.sqrt(
            (self._y - self._target_y) ** 2 +
            (self._z - self._target_z) ** 2
        )
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行动作
        
        Args:
            action: 
                4方向: 0=left, 1=right, 2=up, 3=down
                8方向: 0=left, 1=right, 2=up, 3=down, 
                       4=up_left, 5=up_right, 6=down_left, 7=down_right
        """
        prev_dist = self._distance_to_target()
        
        # 计算步长 (自适应或固定)
        if self.adaptive_step:
            # 距离越远，步长越大
            current_step = min(self.max_step_size, max(self.step_size, prev_dist * 0.3))
        else:
            current_step = self.step_size
        
        # 更新位置
        if self.use_8_direction:
            dy, dz = self.ACTION_VECTORS_8[action]
            dy *= current_step
            dz *= current_step
        else:
            dy = current_step if action == 0 else (-current_step if action == 1 else 0.0)
            dz = current_step if action == 2 else (-current_step if action == 3 else 0.0)
        
        self._y = np.clip(self._y + dy, -1.0, 1.0)
        self._z = np.clip(self._z + dz, -1.0, 1.0)
        self._step_count += 1
        
        # 记录轨迹
        self.trajectory.append((self._y, self._z))
        
        # 发送到物理机械臂 (对角线动作映射为两个连续的基本动作)
        if self.serial_env is not None:
            if self.use_8_direction and action >= 4:
                # 对角线动作: 分解为两个基本动作
                if action == 4:    # up_left
                    self.serial_env.step(0)  # left
                    self.serial_env.step(2)  # up
                elif action == 5:  # up_right
                    self.serial_env.step(1)  # right
                    self.serial_env.step(2)  # up
                elif action == 6:  # down_left
                    self.serial_env.step(0)  # left
                    self.serial_env.step(3)  # down
                elif action == 7:  # down_right
                    self.serial_env.step(1)  # right
                    self.serial_env.step(3)  # down
            else:
                self.serial_env.step(action)
        
        # 计算奖励
        curr_dist = self._distance_to_target()
        reward = (prev_dist - curr_dist) * 1.0
        reward -= 0.01
        
        # 检查是否到达
        reached = curr_dist < self.target_radius
        if reached:
            reward += 10.0
        
        done = reached or (self._step_count >= self.max_steps_per_target)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "position": (self._y, self._z),
            "target": (self._target_y, self._target_z),
        }
        
        return self._get_obs(), reward, done, info
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._y, self._z)


# ============================================================================
# RL 模型加载
# ============================================================================

def load_rl_model(
    model_type: str, 
    device: torch.device, 
    use_8_direction: bool = False,
    use_smooth: bool = False,
    use_optimal: bool = False,
    seq_len: int = 10,
):
    """加载 RL 模型"""
    
    state_dim = 5
    action_dim = 8 if use_8_direction else 4
    
    if use_optimal and use_8_direction:
        # 最优方向 DQN 模型
        from scripts.train_rl_8direction_optimal import MaskedDQN
        
        model = MaskedDQN(state_dim=state_dim, action_dim=action_dim)
        model = model.to(device)
        
        model_paths = [
            _ROOT / "outputs" / "rl_8direction_optimal" / "rl_8direction_optimal.pth",
        ]
        model_name = "8方向最优方向DQN"
    elif use_smooth and use_8_direction:
        # 平滑 8 方向 Transformer 模型
        from scripts.train_rl_8direction_smooth import SequenceTransformerDQN
        
        model = SequenceTransformerDQN(
            state_dim=state_dim, 
            action_dim=action_dim,
            seq_len=seq_len,
        )
        model = model.to(device)
        
        model_paths = [
            _ROOT / "outputs" / "rl_8direction_smooth" / "rl_8direction_smooth_transformer.pth",
        ]
        model_name = "8方向平滑Transformer"
    elif use_8_direction:
        # 8 方向模型
        from scripts.train_rl_8direction import DQN8Direction, TransformerDQN8Direction
        
        if model_type == "transformer":
            model = TransformerDQN8Direction(state_dim=state_dim, action_dim=action_dim)
        else:
            model = DQN8Direction(state_dim=state_dim, action_dim=action_dim)
        
        model = model.to(device)
        
        model_paths = [
            _ROOT / "outputs" / "rl_8direction" / f"rl_8direction_{model_type}.pth",
        ]
        model_name = "8方向"
    else:
        # 4 方向模型
        from scripts.dqn_transformer import TransformerDQN, LightTransformerDQN
        from scripts.dqn_model import DQNNetwork
        
        if model_type == "transformer":
            model = TransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1)
        elif model_type == "light_transformer":
            model = LightTransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1)
        else:
            model = DQNNetwork(state_dim=state_dim, action_dim=action_dim, seq_len=1)
        
        model = model.to(device)
        
        model_paths = [
            _ROOT / "outputs" / "rl_4direction" / f"rl_4direction_{model_type}.pth",
            _ROOT / "outputs" / "rl_control_test" / f"{model_type}_best.pth",
        ]
        model_name = "4方向"
    
    for path in model_paths:
        if path.exists():
            try:
                state = torch.load(path, map_location=device, weights_only=False)
                if isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                model.load_state_dict(state)
                print(f"[RL] 已加载 {model_name} 模型: {path}")
                break
            except Exception as e:
                print(f"[RL] 加载 {path} 失败: {e}")
    
    model.eval()
    return model


# ============================================================================
# 序列控制主循环
# ============================================================================

@dataclass
class SequenceResult:
    """单个序列的控制结果"""
    subject_id: int
    sequence_description: str
    targets: List[str]
    reached_targets: int
    total_targets: int
    total_steps: int
    trajectory: List[Tuple[float, float]]
    target_positions: List[Tuple[float, float]]
    per_target_results: List[Dict]


def run_sequence_control(
    subject_sequence: SubjectSequence,
    rl_model,  # Can be None for rule-based control
    serial_env,
    device: torch.device,
    max_steps_per_target: int = 50,
    target_radius: float = 0.18,
    step_size: float = 0.05,
    max_step_size: float = 0.15,
    adaptive_step: bool = False,
    action_delay: float = 0.3,
    use_8_direction: bool = False,
    use_smooth: bool = False,
    use_optimal: bool = False,
    use_rule: bool = False,
    seq_len: int = 10,
    verbose: bool = True,
) -> SequenceResult:
    """
    运行一个被试的动作序列
    """
    env = SequenceControlEnv(
        serial_env=serial_env,
        target_radius=target_radius,
        step_size=step_size,
        max_step_size=max_step_size,
        adaptive_step=adaptive_step,
        max_steps_per_target=max_steps_per_target,
        use_8_direction=use_8_direction,
    )
    
    if use_8_direction:
        action_names = ["left", "right", "up", "down", 
                        "up_left", "up_right", "down_left", "down_right"]
    else:
        action_names = ["left", "right", "up", "down"]
    
    print("\n" + "="*60)
    print(f"Subject {subject_sequence.subject_id}: {subject_sequence.description}")
    print(f"Sequence: {' → '.join(subject_sequence.sequence_names)}")
    if use_rule:
        print("[Rule Mode] 纯规则控制 - 每步最优方向")
    elif use_optimal:
        print("[Optimal Mode] 强制对齐目标方向")
    elif use_smooth:
        print(f"[Smooth Mode] seq_len={seq_len}")
    print("="*60)
    
    # 导入辅助函数
    if use_optimal or use_rule:
        from scripts.train_rl_8direction_optimal import get_valid_actions_mask, get_optimal_action
    
    # 初始化
    state = env.reset(start_y=0.0, start_z=0.0)
    
    # 平滑模式需要维护状态和动作序列
    if use_smooth:
        from collections import deque
        state_buffer = deque(maxlen=seq_len)
        action_buffer = deque(maxlen=seq_len)
        for _ in range(seq_len):
            state_buffer.append(state)
            action_buffer.append(8)  # 8 = no action
    
    per_target_results = []
    reached_count = 0
    total_steps = 0
    
    # 跳过第一个目标（假设从 center 开始）
    targets = subject_sequence.sequence[1:]  # 排除起点
    
    for i, target in enumerate(targets):
        env.set_target(target.y, target.z)
        
        # 平滑模式更新状态
        if use_smooth:
            state = env._get_obs()
            state_buffer.append(state)
        
        if verbose:
            print(f"\n[Target {i+1}/{len(targets)}] → {target.name} ({target.y:.2f}, {target.z:.2f})")
        
        target_steps = 0
        target_reached = False
        
        # 控制循环
        for step in range(max_steps_per_target):
            # 选择动作
            if use_rule:
                # 纯规则模式：直接选择最优动作
                action = get_optimal_action(env._y, env._z, env._target_y, env._target_z)
            elif use_optimal:
                # 最优方向模式：使用动作掩码
                mask = get_valid_actions_mask(env._y, env._z, env._target_y, env._target_z)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = rl_model(state_tensor, mask_tensor)
                    action = q_values.argmax(dim=1).item()
            elif use_smooth:
                state_seq = np.array(state_buffer)
                action_seq = np.array(action_buffer)
                state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0).to(device)
                action_tensor = torch.tensor(action_seq, dtype=torch.long).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = rl_model(state_tensor, action_tensor)
                    action = q_values.argmax(dim=1).item()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    q_values = rl_model(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            target_steps += 1
            total_steps += 1
            
            if verbose and step % 5 == 0:
                print(f"  Step {step}: {action_names[action]}, pos=({env._y:.2f}, {env._z:.2f}), dist={info['distance']:.3f}")
            
            # 动作延时
            if serial_env is not None:
                time.sleep(action_delay)
            
            state = next_state
            
            # 更新平滑模式的缓冲区
            if use_smooth:
                state_buffer.append(state)
                action_buffer.append(action)
            
            if info["reached"]:
                target_reached = True
                reached_count += 1
                if verbose:
                    print(f"  ✓ 到达目标! (步数: {target_steps})")
                break
        
        if not target_reached and verbose:
            print(f"  ✗ 未到达目标 (最终距离: {info['distance']:.3f})")
        
        per_target_results.append({
            "target_name": target.name,
            "target_pos": (target.y, target.z),
            "reached": target_reached,
            "steps": target_steps,
            "final_distance": info["distance"],
            "final_position": env.position,
        })
    
    # 统计
    print("\n" + "-"*40)
    print(f"序列完成: {reached_count}/{len(targets)} 个目标到达")
    print(f"总步数: {total_steps}")
    print("-"*40)
    
    return SequenceResult(
        subject_id=subject_sequence.subject_id,
        sequence_description=subject_sequence.description,
        targets=subject_sequence.sequence_names,
        reached_targets=reached_count,
        total_targets=len(targets),
        total_steps=total_steps,
        trajectory=env.trajectory,
        target_positions=[(t.y, t.z) for t in subject_sequence.sequence],
        per_target_results=per_target_results,
    )


# ============================================================================
# 可视化: Position vs Time
# ============================================================================

def visualize_sequence_position_vs_time(
    result: SequenceResult,
    output_dir: Path,
    show_plot: bool = False,
):
    """
    可视化序列的 Position vs Time 图
    
    展示:
    - 目标位置变化 (蓝色虚线)
    - 实际位置变化 (红色实线)
    - 到达判定区域 (灰色阴影)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    trajectory = result.trajectory
    time_steps = list(range(len(trajectory)))
    actual_y = [p[0] for p in trajectory]
    actual_z = [p[1] for p in trajectory]
    
    # 构建目标位置时间序列 (分段常量)
    target_y_series = []
    target_z_series = []
    
    current_target_idx = 0
    targets = result.per_target_results
    step_count = 0
    
    for i in range(len(trajectory)):
        if current_target_idx < len(targets):
            target = targets[current_target_idx]
            target_y_series.append(target["target_pos"][0])
            target_z_series.append(target["target_pos"][1])
            
            # 检查是否切换到下一个目标
            step_count += 1
            if step_count > target["steps"] and current_target_idx < len(targets) - 1:
                current_target_idx += 1
                step_count = 0
        else:
            target_y_series.append(targets[-1]["target_pos"][0])
            target_z_series.append(targets[-1]["target_pos"][1])
    
    # 确保长度一致
    while len(target_y_series) < len(trajectory):
        target_y_series.append(target_y_series[-1] if target_y_series else 0)
        target_z_series.append(target_z_series[-1] if target_z_series else 0)
    
    # --- 子图 1: Y 位置 (左右) ---
    ax1 = axes[0, 0]
    ax1.plot(time_steps, target_y_series[:len(time_steps)], 'b--', linewidth=2, label='Target Y', alpha=0.8)
    ax1.plot(time_steps, actual_y, 'r-', linewidth=2, label='Actual Y', alpha=0.8)
    ax1.fill_between(time_steps, target_y_series[:len(time_steps)], actual_y, alpha=0.2, color='orange')
    ax1.set_xlabel('Time Step', fontsize=11)
    ax1.set_ylabel('Y Position (Left ← 0 → Right)', fontsize=11)
    ax1.set_title('Y Position (Horizontal)', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # --- 子图 2: Z 位置 (上下) ---
    ax2 = axes[0, 1]
    ax2.plot(time_steps, target_z_series[:len(time_steps)], 'b--', linewidth=2, label='Target Z', alpha=0.8)
    ax2.plot(time_steps, actual_z, 'r-', linewidth=2, label='Actual Z', alpha=0.8)
    ax2.fill_between(time_steps, target_z_series[:len(time_steps)], actual_z, alpha=0.2, color='orange')
    ax2.set_xlabel('Time Step', fontsize=11)
    ax2.set_ylabel('Z Position (Down ← 0 → Up)', fontsize=11)
    ax2.set_title('Z Position (Vertical)', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)
    
    # --- 子图 3: 2D 轨迹 ---
    ax3 = axes[1, 0]
    # 绘制目标位置
    target_positions = result.target_positions
    for i, (ty, tz) in enumerate(target_positions):
        circle = plt.Circle((ty, tz), 0.15, color='blue', alpha=0.2)
        ax3.add_patch(circle)
        ax3.annotate(result.targets[i], (ty, tz), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    # 绘制轨迹
    ax3.plot(actual_y, actual_z, 'r-', linewidth=2, alpha=0.7, label='Trajectory')
    ax3.scatter([actual_y[0]], [actual_z[0]], color='green', s=100, zorder=5, label='Start')
    ax3.scatter([actual_y[-1]], [actual_z[-1]], color='red', s=100, zorder=5, marker='x', label='End')
    
    ax3.set_xlabel('Y Position (Left ← 0 → Right)', fontsize=11)
    ax3.set_ylabel('Z Position (Down ← 0 → Up)', fontsize=11)
    ax3.set_title('2D Trajectory', fontweight='bold')
    ax3.set_xlim(-1.1, 1.1)
    ax3.set_ylim(-1.1, 1.1)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')
    
    # --- 子图 4: 距离误差 ---
    ax4 = axes[1, 1]
    
    # 计算每步到当前目标的距离
    errors = []
    current_target_idx = 0
    step_count = 0
    
    for i in range(len(trajectory)):
        if current_target_idx < len(targets):
            target = targets[current_target_idx]
            ty, tz = target["target_pos"]
            ay, az = trajectory[i]
            error = np.sqrt((ay - ty)**2 + (az - tz)**2)
            errors.append(error)
            
            step_count += 1
            if step_count > target["steps"] and current_target_idx < len(targets) - 1:
                current_target_idx += 1
                step_count = 0
        else:
            errors.append(errors[-1] if errors else 0)
    
    ax4.plot(time_steps, errors, 'g-', linewidth=2, label='Distance Error')
    ax4.axhline(y=0.15, color='r', linestyle='--', alpha=0.7, label='Target Radius (0.15)')
    ax4.fill_between(time_steps, 0, errors, alpha=0.2, color='green')
    ax4.set_xlabel('Time Step', fontsize=11)
    ax4.set_ylabel('Distance to Current Target', fontsize=11)
    ax4.set_title('Error Convergence', fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # 标记目标切换点
    cumulative_steps = 0
    for i, target_result in enumerate(targets[:-1]):
        cumulative_steps += target_result["steps"]
        ax4.axvline(x=cumulative_steps, color='purple', linestyle=':', alpha=0.5)
    
    # 总标题
    plt.suptitle(
        f'Subject {result.subject_id}: {result.sequence_description}\n'
        f'Reached: {result.reached_targets}/{result.total_targets} targets, '
        f'Total steps: {result.total_steps}',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"subject_{result.subject_id}_position_vs_time.png"
    save_path = output_dir / filename
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print(f"Position vs Time 图已保存: {save_path}")
    return save_path


def visualize_all_subjects_comparison(
    results: List[SequenceResult],
    output_dir: Path,
):
    """
    可视化所有被试的比较图
    """
    n_subjects = len(results)
    fig, axes = plt.subplots(2, (n_subjects + 1) // 2, figsize=(5 * ((n_subjects + 1) // 2), 10))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        # 绘制轨迹
        trajectory = result.trajectory
        actual_y = [p[0] for p in trajectory]
        actual_z = [p[1] for p in trajectory]
        
        # 目标位置
        for ty, tz in result.target_positions:
            circle = plt.Circle((ty, tz), 0.15, color='blue', alpha=0.15)
            ax.add_patch(circle)
        
        # 轨迹线
        ax.plot(actual_y, actual_z, 'r-', linewidth=1.5, alpha=0.7)
        ax.scatter([actual_y[0]], [actual_z[0]], color='green', s=60, zorder=5)
        ax.scatter([actual_y[-1]], [actual_z[-1]], color='red', s=60, zorder=5, marker='x')
        
        # 设置
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Subject {result.subject_id}\n{result.reached_targets}/{result.total_targets} reached',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Y (L↔R)')
        ax.set_ylabel('Z (D↔U)')
    
    # 隐藏多余的子图
    for idx in range(n_subjects, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Multi-Subject Sequence Control Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_dir / "all_subjects_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"所有被试对比图已保存: {save_path}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Multi-Subject Sequence Control")
    
    # 被试选择
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4],
                   help="要测试的被试编号 (默认: 1 2 3 4)")
    
    # 控制参数
    p.add_argument("--max-steps-per-target", type=int, default=50)
    p.add_argument("--target-radius", type=float, default=0.25,
                   help="到达判定半径 (越大越容易到达)")
    p.add_argument("--step-size", type=float, default=0.05,
                   help="基础步长")
    p.add_argument("--adaptive-step", action="store_true",
                   help="使用自适应步长 (距离远时大步，近时小步)")
    p.add_argument("--max-step-size", type=float, default=0.15,
                   help="自适应步长的最大值")
    p.add_argument("--action-delay", type=float, default=0.3)
    
    # 物理控制
    p.add_argument("--physical", action="store_true", help="使用物理机械臂")
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--use-smooth-env", action="store_true", help="使用平滑环境")
    p.add_argument("--move-velocity", type=int, default=80)
    
    # 模型
    p.add_argument("--rl-model", choices=["dqn", "transformer", "light_transformer"],
                   default="dqn")
    p.add_argument("--use-8-direction", action="store_true",
                   help="使用 8 方向 RL 模型 (含对角线)")
    p.add_argument("--use-smooth", action="store_true",
                   help="使用平滑 Transformer 模型 (基于历史序列决策)")
    p.add_argument("--use-optimal", action="store_true",
                   help="使用最优方向 DQN 模型 (强制对齐目标方向)")
    p.add_argument("--use-rule", action="store_true",
                   help="使用纯规则控制 (每步选择最优方向，无 RL)")
    p.add_argument("--seq-len", type=int, default=10,
                   help="平滑模型的历史序列长度")
    p.add_argument("--device", type=str, default="cuda")
    
    # 输出
    p.add_argument("--output-dir", type=Path, 
                   default=_ROOT / "outputs" / "multi_subject_sequence")
    p.add_argument("--show-plots", action="store_true")
    
    # 归位
    p.add_argument("--pre-home", action="store_true")
    p.add_argument("--post-home", action="store_true")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    use_8_direction = getattr(args, 'use_8_direction', False)
    use_smooth = getattr(args, 'use_smooth', False)
    use_optimal = getattr(args, 'use_optimal', False)
    use_rule = getattr(args, 'use_rule', False)
    seq_len = getattr(args, 'seq_len', 10)
    
    if use_rule:
        print("[Mode] 纯规则控制 (每步选择最优方向，无 RL)")
        use_8_direction = True
    elif use_optimal:
        print("[Mode] 最优方向 DQN 模型 (8方向, 强制对齐目标)")
        use_8_direction = True
    elif use_smooth:
        print("[Mode] 平滑 Transformer RL 模型 (8方向, 基于历史序列)")
        use_8_direction = True
    elif use_8_direction:
        print("[Mode] 8 方向 RL 模型 (含对角线)")
    else:
        print("[Mode] 4 方向 RL 模型")
    
    # 加载 RL 模型 (规则模式不需要)
    if use_rule:
        rl_model = None
        print("[Rule] 不加载 RL 模型，使用纯规则控制")
    else:
        rl_model = load_rl_model(
            args.rl_model, device, 
            use_8_direction=use_8_direction,
            use_smooth=use_smooth,
            use_optimal=use_optimal,
            seq_len=seq_len,
        )
    
    # 创建串口环境 (如果物理控制)
    serial_env = None
    if args.physical:
        if args.use_smooth_env:
            from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2
            serial_cfg = SerialConfigV2(
                port=args.serial_port,
                move_velocity=args.move_velocity,
            )
            serial_env = SerialArmEnvV2(serial_cfg, render_mode="human")
        else:
            from serial_arm_env import SerialArmEnv, SerialConfig
            serial_cfg = SerialConfig(port=args.serial_port)
            serial_env = SerialArmEnv(serial_cfg, render_mode="human")
        
        # 预归位
        if args.pre_home:
            import os
            print("[Pre-home] 回到中位...")
            os.system(f"python scripts/serial_go_home.py --port {args.serial_port}")
            time.sleep(1.0)
    
    # 运行每个被试的序列
    all_results = []
    
    for subject_id in args.subjects:
        if subject_id not in SUBJECT_SEQUENCES:
            print(f"警告: Subject {subject_id} 没有预定义的动作序列，跳过")
            continue
        
        subject_seq = SUBJECT_SEQUENCES[subject_id]
        
        try:
            result = run_sequence_control(
                subject_sequence=subject_seq,
                rl_model=rl_model,
                serial_env=serial_env,
                device=device,
                max_steps_per_target=args.max_steps_per_target,
                target_radius=args.target_radius,
                step_size=args.step_size,
                max_step_size=args.max_step_size,
                adaptive_step=args.adaptive_step,
                action_delay=args.action_delay if args.physical else 0.01,
                use_8_direction=use_8_direction,
                use_smooth=use_smooth,
                use_optimal=use_optimal,
                use_rule=use_rule,
                seq_len=seq_len,
                verbose=True,
            )
            
            # 保存单个被试的 Position vs Time 图
            visualize_sequence_position_vs_time(
                result, 
                args.output_dir,
                show_plot=args.show_plots,
            )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"Subject {subject_id} 控制失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 保存所有被试对比图
    if len(all_results) > 1:
        visualize_all_subjects_comparison(all_results, args.output_dir)
    
    # 保存结果 JSON
    results_json = []
    for r in all_results:
        results_json.append({
            "subject_id": r.subject_id,
            "sequence_description": r.sequence_description,
            "targets": r.targets,
            "reached_targets": r.reached_targets,
            "total_targets": r.total_targets,
            "total_steps": r.total_steps,
            "trajectory": [(float(y), float(z)) for y, z in r.trajectory],
            "per_target_results": r.per_target_results,
        })
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存: {args.output_dir}")
    
    # 清理
    if serial_env is not None:
        serial_env.close()
        
        # 结束归位
        if args.post_home:
            import os
            print("\n[Post-home] 回到归位...")
            os.system(f"python scripts/serial_go_return.py --port {args.serial_port}")
    
    # 打印总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    for r in all_results:
        status = "✓" if r.reached_targets == r.total_targets else "△"
        print(f"  Subject {r.subject_id}: {status} {r.reached_targets}/{r.total_targets} targets, {r.total_steps} steps")
    print("="*60)


if __name__ == "__main__":
    main()
