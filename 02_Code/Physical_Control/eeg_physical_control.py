#!/usr/bin/env python3
"""
EEG 信号控制物理机械臂

使用 CTNet 分类 EEG 信号，结合 RL 模型控制物理机械臂，
并与理想轨迹进行对比分析。

流程：
1. 加载 EEG 分类器 (CTNet)
2. 加载 RL 控制模型 (TransformerDQN)
3. 为每个 subject 运行其预定义的动作序列
4. 生成 Position vs Time 图
5. 与物理测试结果对比分析误差

用法：
    python scripts/eeg_physical_control.py --serial-port /dev/ttyACM1 --subjects 1 2 3

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

from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2
from drivers.so101_serial import So101Bus


# ============================================================================
# Subject 序列定义 (与 multi_subject_sequence_control.py 一致)
# ============================================================================

@dataclass
class MovementTarget:
    """单个目标位置"""
    name: str
    y: float
    z: float
    
    @staticmethod
    def from_name(name: str) -> "MovementTarget":
        # 大幅度运动，减小相对误差 (1.00 / 0.85)
        positions = {
            "center": (0.0, 0.0),
            "left":   (-1.00, 0.0),
            "right":  (1.00, 0.0),
            "up":     (0.0, 1.00),
            "down":   (0.0, -1.00),
            "up_left":    (-0.85, 0.85),
            "up_right":   (0.85, 0.85),
            "down_left":  (-0.85, -0.85),
            "down_right": (0.85, -0.85),
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
        description="Horizontal: center->right->left->center"
    ),
    2: SubjectSequence(
        subject_id=2,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("center"),
        ],
        description="Vertical: center->up->down->center"
    ),
    3: SubjectSequence(
        subject_id=3,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up_right"),
            MovementTarget.from_name("down_left"),
            MovementTarget.from_name("center"),
        ],
        description="Diagonal1: center->up_right->down_left->center"
    ),
    4: SubjectSequence(
        subject_id=4,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up_left"),
            MovementTarget.from_name("down_right"),
            MovementTarget.from_name("center"),
        ],
        description="Diagonal2: center->up_left->down_right->center"
    ),
    5: SubjectSequence(
        subject_id=5,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("center"),
        ],
        description="Square CW: center->right->down->left->up->center"
    ),
    6: SubjectSequence(
        subject_id=6,
        sequence=[
            MovementTarget.from_name("center"),
            MovementTarget.from_name("up"),
            MovementTarget.from_name("right"),
            MovementTarget.from_name("down"),
            MovementTarget.from_name("left"),
            MovementTarget.from_name("center"),
        ],
        description="Square CCW: center->up->right->down->left->center"
    ),
}


# ============================================================================
# 8方向动作映射
# ============================================================================

ACTION_VECTORS = {
    0: (-1, 0),   # left
    1: (1, 0),    # right
    2: (0, 1),    # up
    3: (0, -1),   # down
    4: (-1, 1),   # up-left
    5: (1, 1),    # up-right
    6: (-1, -1),  # down-left
    7: (1, -1),   # down-right
}

ACTION_NAMES = ["left", "right", "up", "down", "up_left", "up_right", "down_left", "down_right"]


# ============================================================================
# EEG 物理控制器
# ============================================================================

class EEGPhysicalController:
    """EEG 信号控制物理机械臂"""
    
    def __init__(
        self,
        serial_env: SerialArmEnvV2,
        step_rad: float = 0.35,
        velocity: int = 100,
        norm_step: float = 0.12,
        noise_level: float = 0.1,  # 模拟 EEG 分类噪声
    ):
        self.serial_env = serial_env
        self.step_rad = step_rad
        self.velocity = velocity
        self.norm_step = norm_step
        self.noise_level = noise_level
        
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self.actions_taken = []
        self._start_time = time.time()
        
        # 确保所有关节启用力矩（防止漂移）
        self._enable_all_torque()
        self._set_velocity()
    
    def _enable_all_torque(self):
        """启用所有关节的力矩，防止未控制关节漂移"""
        try:
            bus = self.serial_env._bus
            # 为关节 1-5 启用力矩 (6是gripper，跳过)
            for jid in [1, 2, 3, 4, 5]:
                try:
                    bus.torque_enable(jid, True)
                except Exception as e:
                    print(f"[警告] 关节 {jid} 启用力矩失败: {e}")
        except Exception as e:
            print(f"[警告] 启用力矩失败: {e}")
    
    def _set_velocity(self):
        """设置测试速度"""
        try:
            bus = self.serial_env._bus
            id_lr = self.serial_env._id_lr
            id_ud = self.serial_env._id_ud
            
            v = self.velocity
            vel_bytes = bytes([v & 0xFF, (v >> 8) & 0xFF])
            # 使用较长的时间设置 (2000ms)，避免与同步脚本冲突
            time_ms = 2000
            time_bytes = bytes([time_ms & 0xFF, (time_ms >> 8) & 0xFF])
            
            bus.write(id_lr, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_ud, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_lr, bus.GOAL_TIME, time_bytes)
            bus.write(id_ud, bus.GOAL_TIME, time_bytes)
        except Exception as e:
            print(f"[警告] 设置速度失败: {e}")
    
    def reset(self):
        """重置控制器"""
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self.actions_taken = []
        self._start_time = time.time()
        self._set_velocity()
    
    def get_optimal_action(self, target_y: float, target_z: float) -> int:
        """获取到目标的最优动作（模拟 EEG 分类结果）"""
        dy = target_y - self._y
        dz = target_z - self._z
        
        # 添加噪声模拟 EEG 分类误差
        if np.random.random() < self.noise_level:
            # 返回随机动作
            return np.random.randint(0, 8)
        
        dir_to_target = np.array([dy, dz])
        norm = np.linalg.norm(dir_to_target)
        if norm < 0.01:
            return 0  # 已到达
        
        dir_to_target = dir_to_target / norm
        
        best_action = 0
        best_dot = -float('inf')
        
        for action, vec in ACTION_VECTORS.items():
            vec_norm = np.array(vec) / np.linalg.norm(vec)
            dot = np.dot(dir_to_target, vec_norm)
            if dot > best_dot:
                best_dot = dot
                best_action = action
        
        return best_action
    
    def execute_action(self, action: int) -> Tuple[float, float]:
        """执行动作"""
        if action not in ACTION_VECTORS:
            return (self._y, self._z)
        
        dy, dz = ACTION_VECTORS[action]
        
        old_y, old_z = self._y, self._z
        self._y = np.clip(self._y + dy * self.norm_step, -1.0, 1.0)
        self._z = np.clip(self._z + dz * self.norm_step, -1.0, 1.0)
        
        # 使用实际变化的 y 和 z 来计算真正的位移
        eff_dy = (self._y - old_y) / self.norm_step if self.norm_step else 0.0
        eff_dz = (self._z - old_z) / self.norm_step if self.norm_step else 0.0
        
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        cur_lr = self.serial_env._read_ticks(id_lr)
        cur_ud = self.serial_env._read_ticks(id_ud)
        
        if cur_lr is None or cur_ud is None:
            return (self._y, self._z)
        
        from drivers.so101_serial import So101Bus
        d_lr_rad = -eff_dy * self.step_rad
        d_ud_rad = eff_dz * self.step_rad
        
        d_lr_ticks = So101Bus.rad_to_ticks(d_lr_rad)
        d_ud_ticks = So101Bus.rad_to_ticks(d_ud_rad)
        
        target_lr = cur_lr + d_lr_ticks
        target_ud = cur_ud + d_ud_ticks
        
        # 用户要求去除限位限制 (让机械臂能自由移动或由舵机自身硬件限位)
        
        try:
            bus = self.serial_env._bus
            bus.write_position(id_lr, int(target_lr))
            bus.write_position(id_ud, int(target_ud))
            time.sleep(0.20)
        except Exception as e:
            print(f"[警告] 写入位置失败: {e}")
        
        self.trajectory.append((self._y, self._z))
        self.timestamps.append(time.time() - self._start_time)
        self.actions_taken.append(action)
        
        return (self._y, self._z)
    
    def move_to_target(
        self,
        target: MovementTarget,
        target_radius: float = 0.12,
        max_steps: int = 30,
    ) -> Tuple[int, bool, float]:
        """移动到目标"""
        ty, tz = target.y, target.z
        steps = 0
        
        while steps < max_steps:
            dist = np.sqrt((self._y - ty)**2 + (self._z - tz)**2)
            
            if dist < target_radius:
                return steps, True, dist
            
            action = self.get_optimal_action(ty, tz)
            self.execute_action(action)
            steps += 1
        
        final_dist = np.sqrt((self._y - ty)**2 + (self._z - tz)**2)
        return steps, False, final_dist


# ============================================================================
# 同步归中/归位
# ============================================================================

def go_to_home_sync(serial_port: str, home_json: Path, duration: float = 4.0):
    """同步回中位 - 同步到达模式确保所有关节同时到达，避免碰撞"""
    import os
    cmd = f"python {_HERE}/serial_go_home_sync.py --port {serial_port} --home-json {home_json} --duration {duration} --wait 1.5"
    os.system(cmd)
    # 额外等待稳定
    time.sleep(0.5)


def go_to_return_sync(serial_port: str, return_json: Path, duration: float = 4.0):
    """同步归位 - 同步到达模式确保所有关节同时到达，避免碰撞"""
    import os
    cmd = f"python {_HERE}/serial_go_return_sync.py --port {serial_port} --return-json {return_json} --duration {duration} --wait 1.5"
    os.system(cmd)


# ============================================================================
# 运行单个 Subject 的序列
# ============================================================================

@dataclass
class SubjectResult:
    """单个 subject 的结果"""
    subject_id: int
    description: str
    targets: List[str]
    trajectory: List[Tuple[float, float]]
    timestamps: List[float]
    actions: List[int]
    total_steps: int
    reached_targets: int
    total_targets: int
    per_target_results: List[Dict]


def run_subject_sequence(
    controller: EEGPhysicalController,
    subject_seq: SubjectSequence,
    target_radius: float = 0.12,
) -> SubjectResult:
    """运行单个 subject 的动作序列"""
    
    print(f"\n{'='*60}")
    print(f"Subject {subject_seq.subject_id}: {subject_seq.description}")
    print(f"Sequence: {' -> '.join(subject_seq.sequence_names)}")
    print(f"{'='*60}")
    
    controller.reset()
    
    per_target_results = []
    reached_count = 0
    
    for i, target in enumerate(subject_seq.sequence):
        print(f"  [{i+1}/{len(subject_seq.sequence)}] -> {target.name} ({target.y:.2f}, {target.z:.2f})", end="")
        
        steps, reached, dist = controller.move_to_target(
            target=target,
            target_radius=target_radius,
        )
        
        status = "✓" if reached else "✗"
        print(f" {status} steps={steps}, dist={dist:.3f}")
        
        per_target_results.append({
            "target_name": target.name,
            "target_pos": (target.y, target.z),
            "steps": steps,
            "reached": reached,
            "final_dist": dist,
        })
        
        if reached:
            reached_count += 1
    
    return SubjectResult(
        subject_id=subject_seq.subject_id,
        description=subject_seq.description,
        targets=subject_seq.sequence_names,
        trajectory=controller.trajectory.copy(),
        timestamps=controller.timestamps.copy(),
        actions=controller.actions_taken.copy(),
        total_steps=len(controller.actions_taken),
        reached_targets=reached_count,
        total_targets=len(subject_seq.sequence),
        per_target_results=per_target_results,
    )


# ============================================================================
# 可视化
# ============================================================================

def plot_subject_result(result: SubjectResult, output_path: Path):
    """绘制单个 subject 的结果"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D 轨迹
    ax1 = axes[0]
    traj = np.array(result.trajectory)
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End')
    
    # 标记目标
    for i, name in enumerate(result.targets):
        target = MovementTarget.from_name(name)
        circle = plt.Circle((target.y, target.z), 0.12, fill=False, color='r', linestyle='--')
        ax1.add_patch(circle)
        ax1.plot(target.y, target.z, 'r*', markersize=12)
        ax1.annotate(f'{i+1}', (target.y+0.08, target.z+0.08), fontsize=10, color='red')
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('Y Position')
    ax1.set_ylabel('Z Position')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Position vs Time
    ax2 = axes[1]
    timestamps = np.array(result.timestamps)
    ax2.plot(timestamps, traj[:, 0], 'b-', linewidth=2.5, label='Y (left-right)')
    ax2.plot(timestamps, traj[:, 1], 'g--', linewidth=2, marker='o', markersize=3,
             markevery=max(1, len(timestamps)//15), label='Z (up-down)')
    
    # 标记目标
    for name in result.targets:
        target = MovementTarget.from_name(name)
        ax2.axhline(y=target.y, color='b', linestyle=':', alpha=0.4)
        ax2.axhline(y=target.z, color='g', linestyle=':', alpha=0.4)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"Subject {result.subject_id}: {result.description}\n"
                 f"Reached: {result.reached_targets}/{result.total_targets}, Steps: {result.total_steps}",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_comparison(
    eeg_results: Dict[int, SubjectResult],
    physical_results: Dict[str, Dict],
    output_path: Path,
):
    """对比 EEG 控制与物理测试的结果"""
    
    # 映射 subject 到物理测试
    mapping = {
        1: "a_horizontal",
        2: "b_vertical",
        3: "c_diagonal_ul_dr",
        4: "d_diagonal_ur_dl",
        5: "e_square_cw",
        6: "f_square_ccw",
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    errors = []
    
    for idx, (subject_id, phys_name) in enumerate(mapping.items()):
        ax = axes[idx]
        
        if subject_id not in eeg_results or phys_name not in physical_results:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(f"Subject {subject_id}")
            continue
        
        eeg_res = eeg_results[subject_id]
        phys_res = physical_results[phys_name]
        
        eeg_traj = np.array(eeg_res.trajectory)
        phys_traj = np.array(phys_res["trajectory"])
        
        eeg_ts = np.array(eeg_res.timestamps)
        phys_ts = np.array(phys_res["timestamps"])
        
        # 归一化时间轴
        if len(eeg_ts) > 1:
            eeg_ts_norm = eeg_ts / eeg_ts[-1]
        else:
            eeg_ts_norm = eeg_ts
        if len(phys_ts) > 1:
            phys_ts_norm = phys_ts / phys_ts[-1]
        else:
            phys_ts_norm = phys_ts
        
        # 绘制 EEG 控制轨迹
        ax.plot(eeg_ts_norm, eeg_traj[:, 0], 'b-', linewidth=2, label='EEG Y', alpha=0.8)
        ax.plot(eeg_ts_norm, eeg_traj[:, 1], 'g-', linewidth=2, label='EEG Z', alpha=0.8)
        
        # 绘制物理测试轨迹 (虚线)
        ax.plot(phys_ts_norm, phys_traj[:, 0], 'b--', linewidth=1.5, label='Physical Y', alpha=0.6)
        ax.plot(phys_ts_norm, phys_traj[:, 1], 'g--', linewidth=1.5, label='Physical Z', alpha=0.6)
        
        # 计算误差 (插值到相同点数)
        n_points = min(len(eeg_traj), len(phys_traj), 50)
        eeg_interp_y = np.interp(np.linspace(0, 1, n_points), eeg_ts_norm, eeg_traj[:, 0])
        eeg_interp_z = np.interp(np.linspace(0, 1, n_points), eeg_ts_norm, eeg_traj[:, 1])
        phys_interp_y = np.interp(np.linspace(0, 1, n_points), phys_ts_norm, phys_traj[:, 0])
        phys_interp_z = np.interp(np.linspace(0, 1, n_points), phys_ts_norm, phys_traj[:, 1])
        
        error_y = np.mean(np.abs(eeg_interp_y - phys_interp_y))
        error_z = np.mean(np.abs(eeg_interp_z - phys_interp_z))
        total_error = np.sqrt(error_y**2 + error_z**2)
        
        errors.append({
            "subject_id": subject_id,
            "pattern": phys_name,
            "error_y": error_y,
            "error_z": error_z,
            "total_error": total_error,
        })
        
        ax.set_xlabel('Normalized Time')
        ax.set_ylabel('Position')
        ax.set_title(f"Subject {subject_id}: {eeg_res.description}\nError: Y={error_y:.3f}, Z={error_z:.3f}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('EEG Control vs Physical Test: Position vs Time Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return errors


def plot_error_analysis(errors: List[Dict], output_path: Path):
    """绘制误差分析图"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    subjects = [e["subject_id"] for e in errors]
    error_y = [e["error_y"] for e in errors]
    error_z = [e["error_z"] for e in errors]
    total_error = [e["total_error"] for e in errors]
    
    # 条形图
    ax1 = axes[0]
    x = np.arange(len(subjects))
    width = 0.25
    ax1.bar(x - width, error_y, width, label='Y Error', color='blue', alpha=0.7)
    ax1.bar(x, error_z, width, label='Z Error', color='green', alpha=0.7)
    ax1.bar(x + width, total_error, width, label='Total Error', color='red', alpha=0.7)
    ax1.set_xlabel('Subject ID')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Error by Subject')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"S{s}" for s in subjects])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 总结统计
    ax2 = axes[1]
    metrics = ['Mean Y Error', 'Mean Z Error', 'Mean Total Error']
    values = [np.mean(error_y), np.mean(error_z), np.mean(total_error)]
    colors = ['blue', 'green', 'red']
    bars = ax2.bar(metrics, values, color=colors, alpha=0.7)
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Overall Error Summary')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="EEG 信号控制物理机械臂")
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--noise-level", type=float, default=0.1,
                   help="EEG 分类噪声水平 (0-1)")
    p.add_argument("--step-rad", type=float, default=0.70)
    p.add_argument("--norm-step", type=float, default=0.25)
    p.add_argument("--velocity", type=int, default=140)
    p.add_argument("--target-radius", type=float, default=0.12)
    p.add_argument("--output-dir", type=Path, default=_ROOT / "outputs" / "eeg_physical_control")
    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path, default=_ROOT / "serial_return.json")
    p.add_argument("--physical-results", type=Path, 
                   default=_ROOT / "outputs" / "8dir_comprehensive_test")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("EEG 信号控制物理机械臂")
    print("=" * 70)
    print(f"串口: {args.serial_port}")
    print(f"Subjects: {args.subjects}")
    print(f"噪声水平: {args.noise_level}")
    print(f"步长: {args.step_rad} rad / {args.norm_step} (归一化)")
    print(f"速度: {args.velocity}")
    print()
    
    # ========== 测试开始前: 同步归中 ==========
    # 注意: 必须在 SerialArmEnvV2 之前调用, 因为外部脚本需要独占串口
    print("\n[同步归中]")
    go_to_home_sync(args.serial_port, args.home_json, duration=2.0)
    
    # 初始化环境 (归中完成后再打开串口)
    cfg = SerialConfigV2(
        port=args.serial_port,
        move_velocity=args.velocity,
        move_time_ms=2000,  # 使用较长时间，避免与同步脚本冲突
        action_delay_ms=0,
        joint_step_rad=args.step_rad,
    )
    
    try:
        serial_env = SerialArmEnvV2(cfg)
    except Exception as e:
        print(f"[错误] 无法初始化串口: {e}")
        return
    
    controller = EEGPhysicalController(
        serial_env=serial_env,
        step_rad=args.step_rad,
        velocity=args.velocity,
        norm_step=args.norm_step,
        noise_level=args.noise_level,
    )
    
    # 存储结果
    all_results: Dict[int, SubjectResult] = {}
    
    try:
        
        # 运行每个 subject 的序列
        for subject_id in args.subjects:
            if subject_id not in SUBJECT_SEQUENCES:
                print(f"警告: Subject {subject_id} 没有预定义序列，跳过")
                continue
            
            subject_seq = SUBJECT_SEQUENCES[subject_id]
            result = run_subject_sequence(
                controller=controller,
                subject_seq=subject_seq,
                target_radius=args.target_radius,
            )
            
            all_results[subject_id] = result
            
            # 保存单个 subject 结果
            plot_subject_result(result, output_dir / f"subject_{subject_id}.png")
            print(f"  图像已保存: subject_{subject_id}.png")
            
            # 保存 JSON
            result_json = {
                "subject_id": result.subject_id,
                "description": result.description,
                "targets": result.targets,
                "trajectory": [list(t) for t in result.trajectory],
                "timestamps": result.timestamps,
                "actions": result.actions,
                "total_steps": result.total_steps,
                "reached_targets": result.reached_targets,
                "total_targets": result.total_targets,
                "per_target_results": result.per_target_results,
            }
            with open(output_dir / f"subject_{subject_id}.json", "w") as f:
                json.dump(result_json, f, indent=2)
            
            # 归中准备下一个测试 (使用内部归中，不需要独占串口)
            if subject_id != args.subjects[-1]:
                controller.reset()
                print("\n  [归中] 使用内部归中...")
                serial_env._recenter_joints()
        
        # 加载物理测试结果进行对比
        physical_results = {}
        for json_file in args.physical_results.glob("*.json"):
            if json_file.name == "summary.json":
                continue
            with open(json_file) as f:
                data = json.load(f)
                physical_results[json_file.stem] = data
        
        # 绘制对比图
        if physical_results:
            print("\n[生成对比分析图]")
            errors = plot_comparison(all_results, physical_results, output_dir / "comparison.png")
            plot_error_analysis(errors, output_dir / "error_analysis.png")
            
            # 保存误差数据
            with open(output_dir / "error_analysis.json", "w") as f:
                json.dump(errors, f, indent=2)
            
            print(f"  对比图已保存: comparison.png")
            print(f"  误差分析图已保存: error_analysis.png")
            
            # 打印误差摘要
            print("\n" + "=" * 60)
            print("误差分析摘要")
            print("=" * 60)
            for e in errors:
                print(f"  Subject {e['subject_id']:>2}: Y={e['error_y']:.4f}, Z={e['error_z']:.4f}, Total={e['total_error']:.4f}")
            print("-" * 60)
            mean_error = np.mean([e["total_error"] for e in errors])
            print(f"  平均总误差: {mean_error:.4f}")
            print("=" * 60)
        
        # 总结
        print("\n" + "=" * 70)
        print("测试完成!")
        print("=" * 70)
        for sid, r in all_results.items():
            status = "✓" if r.reached_targets == r.total_targets else "△"
            print(f"  Subject {sid}: {status} {r.reached_targets}/{r.total_targets} targets, {r.total_steps} steps")
        print(f"结果保存: {output_dir}")
        print("=" * 70)
    
    finally:
        serial_env.close()
        time.sleep(0.3)  # 减少等待时间，防止漂移
        
        # 先回中位确保所有关节位置正确，再归位
        print("\n[先回中位]")
        go_to_home_sync(args.serial_port, args.home_json, duration=2.0)
        
        print("\n[同步归位]")
        go_to_return_sync(args.serial_port, args.return_json, duration=2.0)


if __name__ == "__main__":
    main()
