#!/usr/bin/env python3
"""
8 方向综合运动测试

测试序列：
a. 水平运动 (左->右->中)
b. 垂直运动 (上->下->中)
c. 左上到右下斜角 (左上->右下->中)
d. 右上到左下斜角 (右上->左下->中)
e. 顺时针正方形 (右->下->左->上->中)
f. 逆时针正方形 (上->右->下->左->中) + 归位

用法：
    python scripts/test_8dir_comprehensive.py --serial-port /dev/ttyACM1

创建时间: 2026-03-03
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2
from drivers.so101_serial import So101Bus, So101Map


# ============================================================================
# 8 方向动作定义
# ============================================================================

ACTION_NAMES = {
    0: "left",
    1: "right", 
    2: "up",
    3: "down",
    4: "up_left",
    5: "up_right",
    6: "down_left",
    7: "down_right",
}

ACTION_VECTORS = {
    0: (-1.0, 0.0),   # left
    1: (1.0, 0.0),    # right
    2: (0.0, 1.0),    # up
    3: (0.0, -1.0),   # down
    4: (-0.707, 0.707),   # up_left
    5: (0.707, 0.707),    # up_right
    6: (-0.707, -0.707),  # down_left
    7: (0.707, -0.707),   # down_right
}

NAME_TO_ACTION = {v: k for k, v in ACTION_NAMES.items()}


# ============================================================================
# 测试序列定义
# ============================================================================

@dataclass
class TestPattern:
    """测试模式"""
    name: str
    description: str
    targets: List[Tuple[float, float]]


# 定义所有测试模式 - 大幅度运动，减小相对误差
TEST_PATTERNS = {
    "a_horizontal": TestPattern(
        name="a_horizontal",
        description="Horizontal (Left->Right->Center)",
        targets=[(-1.00, 0.0), (1.00, 0.0), (0.0, 0.0)],
    ),
    "b_vertical": TestPattern(
        name="b_vertical", 
        description="Vertical (Up->Down->Center)",
        targets=[(0.0, 1.00), (0.0, -1.00), (0.0, 0.0)],
    ),
    "c_diagonal_ul_dr": TestPattern(
        name="c_diagonal_ul_dr",
        description="Diagonal (UpperLeft->LowerRight)",
        targets=[(-0.85, 0.85), (0.85, -0.85), (0.0, 0.0)],
    ),
    "d_diagonal_ur_dl": TestPattern(
        name="d_diagonal_ur_dl",
        description="Diagonal (UpperRight->LowerLeft)",
        targets=[(0.85, 0.85), (-0.85, -0.85), (0.0, 0.0)],
    ),
    "e_square_cw": TestPattern(
        name="e_square_cw",
        description="Square (Clockwise)",
        targets=[(0.85, 0.0), (0.85, -0.85), (-0.85, -0.85), (-0.85, 0.85), (0.85, 0.85), (0.0, 0.0)],
    ),
    "f_square_ccw": TestPattern(
        name="f_square_ccw",
        description="Square (Counter-Clockwise)",
        targets=[(0.0, 0.85), (0.85, 0.85), (0.85, -0.85), (-0.85, -0.85), (-0.85, 0.85), (0.0, 0.0)],
    ),
}


# ============================================================================
# 匀速运动控制器
# ============================================================================

class SmoothMotionController:
    """
    匀速运动控制器 - 测试用
    
    测试速度: 100 (所有关节统一)
    """
    
    TEST_VELOCITY = 100  # 测试速度
    
    def __init__(
        self,
        serial_env: SerialArmEnvV2,
        step_rad: float = 0.35,  # 每步物理移动量 (弧度)
        velocity: int = 100,     # 测试速度
        norm_step: float = 0.12, # 归一化步长 (更多步数)
        use_soft_motion: bool = True,
    ):
        self.serial_env = serial_env
        self.step_rad = step_rad
        self.velocity = velocity
        self.norm_step = norm_step
        self.use_soft_motion = use_soft_motion
        
        self._y = 0.0
        self._z = 0.0
        
        self.trajectory: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.timestamps: List[float] = [0.0]
        self._start_time = time.time()
        
        # 确保所有关节启用力矩（防止漂移）
        self._enable_all_torque()
        # 初始化时设置控制关节的测试速度
        self._set_test_velocity()
    
    def _enable_all_torque(self):
        """启用所有关节的力矩，防止未控制关节漂移"""
        try:
            bus = self.serial_env._bus
            for jid in [1, 2, 3, 4, 5]:
                try:
                    bus.torque_enable(jid, True)
                except Exception:
                    pass
        except Exception:
            pass
    
    def _set_test_velocity(self):
        """设置测试速度 (两个控制关节统一速度)"""
        try:
            bus = self.serial_env._bus
            id_lr = self.serial_env._id_lr
            id_ud = self.serial_env._id_ud
            
            v = self.velocity
            vel_bytes = bytes([v & 0xFF, (v >> 8) & 0xFF])
            # 使用较长的时间设置 (2000ms)，避免与同步脚本冲突
            time_ms = 2000
            time_bytes = bytes([time_ms & 0xFF, (time_ms >> 8) & 0xFF])
            
            # 设置两个控制关节的速度
            bus.write(id_lr, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_ud, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_lr, bus.GOAL_TIME, time_bytes)
            bus.write(id_ud, bus.GOAL_TIME, time_bytes)
            
            print(f"[测试速度] velocity={v}, time={time_ms}ms")
        except Exception as e:
            print(f"[警告] 设置速度失败: {e}")
    
    def reset(self):
        """重置控制器状态"""
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self._start_time = time.time()
        self._set_test_velocity()
    
    def execute_action(self, action: int) -> Tuple[float, float]:
        """
        执行单个动作 - 全程匀速
        """
        if action not in ACTION_VECTORS:
            return (self._y, self._z)
        
        old_y, old_z = self._y, self._z
        dy, dz = ACTION_VECTORS[action]
        
        # 使用实际变化的 y 和 z 来计算真正的位移
        eff_dy = (self._y - old_y) / self.norm_step if self.norm_step else 0.0
        eff_dz = (self._z - old_z) / self.norm_step if self.norm_step else 0.0
        
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        cur_lr = self.serial_env._read_ticks(id_lr)
        cur_ud = self.serial_env._read_ticks(id_ud)
        
        if cur_lr is None or cur_ud is None:
            return (self._y, self._z)
        
        d_lr_rad = -eff_dy * self.step_rad
        d_ud_rad = eff_dz * self.step_rad
        
        d_lr_ticks = So101Bus.rad_to_ticks(d_lr_rad)
        d_ud_ticks = So101Bus.rad_to_ticks(d_ud_rad)
        
        target_lr = cur_lr + d_lr_ticks
        target_ud = cur_ud + d_ud_ticks
        
        # 用户要求去除限位限制 (让机械臂能自由移动或由舵机自身硬件限位)
        
        try:
            bus = self.serial_env._bus
            # 同时写入两个关节位置
            bus.write_position(id_lr, int(target_lr))
            bus.write_position(id_ud, int(target_ud))
            # 等待舵机完成移动 (step_rad=0.35 约需 0.2s)
            time.sleep(0.20)
        except Exception as e:
            print(f"[警告] 写入位置失败: {e}")
        
        self.trajectory.append((self._y, self._z))
        self.timestamps.append(time.time() - self._start_time)
        
        return (self._y, self._z)
    
    def move_to_target(
        self, 
        target: Tuple[float, float], 
        target_radius: float = 0.12,
        max_steps: int = 30,
    ) -> Tuple[int, bool, float]:
        """
        移动到目标点 - 全程匀速
        
        Args:
            target: 目标坐标
            target_radius: 到达判定半径
            max_steps: 最大步数
        
        Returns:
            (steps, reached, final_dist)
        """
        ty, tz = target
        steps = 0
        
        while steps < max_steps:
            dist = np.sqrt((self._y - ty)**2 + (self._z - tz)**2)
            
            if dist < target_radius:
                return steps, True, dist
            
            # 选择最优动作
            dir_to_target = np.array([ty - self._y, tz - self._z])
            dir_to_target = dir_to_target / (np.linalg.norm(dir_to_target) + 1e-8)
            
            best_action = 0
            best_dot = -float('inf')
            
            for action, vec in ACTION_VECTORS.items():
                dot = np.dot(dir_to_target, vec)
                if dot > best_dot:
                    best_dot = dot
                    best_action = action
            
            # 匀速运动 (execute_action 内部已有延时)
            self.execute_action(best_action)
            steps += 1
        
        final_dist = np.sqrt((self._y - ty)**2 + (self._z - tz)**2)
        return steps, False, final_dist
    
    def return_to_center(self):
        """
        回到中位 - 使用测试速度 (velocity 100)
        
        用于测试过程中的归中，不使用同步归中
        """
        print("\n  [归中] 使用测试速度...")
        
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        # 获取中心位置
        center_lr = self.serial_env._joint_centers.get(id_lr)
        center_ud = self.serial_env._joint_centers.get(id_ud)
        
        if center_lr is None or center_ud is None:
            print("  [警告] 无法获取中心位置")
            return
        
        try:
            bus = self.serial_env._bus
            # 同时移动到中心位置
            bus.write_position(id_lr, center_lr)
            bus.write_position(id_ud, center_ud)
            # 等待移动完成 (wrist_flex 需要更多时间)
            time.sleep(1.5)
        except Exception as e:
            print(f"  [警告] 归中失败: {e}")
        
        # 重置内部状态
        self._y = 0.0
        self._z = 0.0
        self.trajectory.append((0.0, 0.0))
        self.timestamps.append(time.time() - self._start_time)
        print("  [归中] 完成")


# ============================================================================
# 归位函数 (同步模式，用于测试前后)
# ============================================================================

def go_to_home(serial_port: str, home_json: Path, duration: float = 4.0):
    """
    回到中位 - 同步到达模式
    
    所有关节同时到达目标位置，确保运动协调避免碰撞
    """
    if not home_json.exists():
        print(f"[警告] 中位文件不存在: {home_json}")
        return
    
    print("\n[同步回中位]")
    cmd = f"python {_HERE}/serial_go_home_sync.py --port {serial_port} --home-json {home_json} --duration {duration} --wait 1.5"
    os.system(cmd)
    # 额外等待稳定
    time.sleep(0.5)


def go_to_return(serial_port: str, return_json: Path, duration: float = 4.0):
    """
    回到归位 - 同步到达模式
    
    所有关节同时到达目标位置，确保运动协调避免碰撞
    """
    if not return_json.exists():
        print(f"[警告] 归位文件不存在: {return_json}")
        return
    
    print("\n[同步归位]")
    cmd = f"python {_HERE}/serial_go_return_sync.py --port {serial_port} --return-json {return_json} --duration {duration} --wait 1.5"
    os.system(cmd)


# ============================================================================
# 单模式测试
# ============================================================================

def run_single_pattern(
    controller: SmoothMotionController,
    pattern: TestPattern,
    target_radius: float = 0.12,
    repeat: int = 2,  # 每个动作重复次数
) -> Dict:
    """
    运行单个测试模式
    
    Args:
        repeat: 重复次数（默认做两遍）
    """
    print(f"\n{'='*60}")
    print(f"测试: {pattern.description} (重复 {repeat} 次)")
    print(f"目标序列: {pattern.targets}")
    print(f"{'='*60}")
    
    controller.reset()
    
    results = {
        "name": pattern.name,
        "description": pattern.description,
        "targets": pattern.targets,
        "repeat": repeat,
        "per_target": [],
        "total_steps": 0,
        "reached": 0,
    }
    
    total_targets = len(pattern.targets) * repeat
    target_count = 0
    
    for rep in range(repeat):
        if repeat > 1:
            print(f"  --- 第 {rep+1}/{repeat} 遍 ---")
        
        for i, target in enumerate(pattern.targets):
            target_count += 1
            
            print(f"  [{target_count}/{total_targets}] -> ({target[0]:.2f}, {target[1]:.2f})", end="")
            
            steps, reached, dist = controller.move_to_target(
                target, 
                target_radius=target_radius,
            )
            
            status = "✓" if reached else "✗"
            print(f" {status} steps={steps}, dist={dist:.3f}")
            
            results["per_target"].append({
                "target": target,
                "steps": steps,
                "reached": reached,
                "final_dist": dist,
            })
            results["total_steps"] += steps
            if reached:
                results["reached"] += 1
    
    results["trajectory"] = controller.trajectory.copy()
    results["timestamps"] = controller.timestamps.copy()
    
    return results


# ============================================================================
# 可视化
# ============================================================================

def plot_single_pattern(results: Dict, output_path: Path):
    """绘制单个模式的图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 2D 轨迹
    ax1 = axes[0]
    traj = np.array(results["trajectory"])
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    ax1.plot(traj[-1, 0], traj[-1, 1], 'rs', markersize=10, label='End')
    
    for i, t in enumerate(results["targets"]):
        circle = plt.Circle(t, 0.12, fill=False, color='r', linestyle='--', linewidth=1.5)
        ax1.add_patch(circle)
        ax1.plot(t[0], t[1], 'r*', markersize=12)
        ax1.annotate(f'{i+1}', (t[0]+0.08, t[1]+0.08), fontsize=10, color='red')
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_xlabel('Y Position (left-right)', fontsize=11)
    ax1.set_ylabel('Z Position (up-down)', fontsize=11)
    ax1.set_title('2D Trajectory', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Position vs Time
    ax2 = axes[1]
    timestamps = np.array(results["timestamps"])
    # 使用不同线型和标记，避免重叠时看不清
    ax2.plot(timestamps, traj[:, 0], 'b-', linewidth=2.5, label='Y (left-right)')
    ax2.plot(timestamps, traj[:, 1], 'g--', linewidth=2, marker='o', markersize=3, 
             markevery=max(1, len(timestamps)//15), label='Z (up-down)')
    
    # 标记目标
    for t in results["targets"]:
        ax2.axhline(y=t[0], color='b', linestyle=':', alpha=0.5, linewidth=1.5)
        ax2.axhline(y=t[1], color='g', linestyle=':', alpha=0.5, linewidth=1.5)
    
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Position', fontsize=11)
    ax2.set_title('Position vs Time', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f"{results['description']}\nTotal Steps: {results['total_steps']}", fontsize=13)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_patterns(all_results: Dict[str, Dict], output_path: Path):
    """绘制所有模式的 Position vs Time 对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    pattern_order = ["a_horizontal", "b_vertical", "c_diagonal_ul_dr", 
                     "d_diagonal_ur_dl", "e_square_cw", "f_square_ccw"]
    
    labels = {
        "a_horizontal": "a. Horizontal",
        "b_vertical": "b. Vertical",
        "c_diagonal_ul_dr": "c. UL to DR Diagonal",
        "d_diagonal_ur_dl": "d. UR to DL Diagonal",
        "e_square_cw": "e. Clockwise Square",
        "f_square_ccw": "f. Counter-CW Square",
    }
    
    for idx, name in enumerate(pattern_order):
        ax = axes[idx]
        
        if name not in all_results:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center', fontsize=14)
            ax.set_title(labels.get(name, name))
            continue
        
        results = all_results[name]
        traj = np.array(results["trajectory"])
        timestamps = np.array(results["timestamps"])
        
        # 使用不同线型和标记，避免斜角运动时Y/Z重叠看不清
        ax.plot(timestamps, traj[:, 0], 'b-', linewidth=2.5, label='Y (left-right)')
        ax.plot(timestamps, traj[:, 1], 'g--', linewidth=2, marker='o', markersize=2,
                markevery=max(1, len(timestamps)//10), label='Z (up-down)')
        
        for t in results["targets"]:
            ax.axhline(y=t[0], color='b', linestyle=':', alpha=0.4)
            ax.axhline(y=t[1], color='g', linestyle=':', alpha=0.4)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position')
        ax.set_title(f"{labels.get(name, name)} ({results['total_steps']} steps)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
    
    plt.suptitle("8-Direction Comprehensive Test: Position vs Time", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n综合图已保存: {output_path}")


def plot_all_trajectories(all_results: Dict[str, Dict], output_path: Path):
    """绘制所有模式的 2D 轨迹对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    pattern_order = ["a_horizontal", "b_vertical", "c_diagonal_ul_dr", 
                     "d_diagonal_ur_dl", "e_square_cw", "f_square_ccw"]
    
    labels = {
        "a_horizontal": "a. Horizontal",
        "b_vertical": "b. Vertical",
        "c_diagonal_ul_dr": "c. UL-DR Diagonal",
        "d_diagonal_ur_dl": "d. UR-DL Diagonal",
        "e_square_cw": "e. CW Square",
        "f_square_ccw": "f. CCW Square",
    }
    
    for idx, name in enumerate(pattern_order):
        ax = axes[idx]
        
        if name not in all_results:
            ax.text(0.5, 0.5, "No Data", ha='center', va='center')
            ax.set_title(labels.get(name, name))
            continue
        
        results = all_results[name]
        traj = np.array(results["trajectory"])
        
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        
        for i, t in enumerate(results["targets"]):
            circle = plt.Circle(t, 0.12, fill=False, color='r', linestyle='--', linewidth=1.5)
            ax.add_patch(circle)
            ax.plot(t[0], t[1], 'r*', markersize=10)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.set_title(f"{labels.get(name, name)}")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.suptitle("8-Direction Comprehensive Test: 2D Trajectories", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"轨迹图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="8 方向综合运动测试")
    
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--step-rad", type=float, default=0.70,
                   help="每步弧度 (默认 0.70)")
    p.add_argument("--norm-step", type=float, default=0.25,
                   help="归一化步长 (默认 0.25)")
    p.add_argument("--velocity", type=int, default=140,
                   help="测试运动速度 (默认 140)")
    p.add_argument("--no-soft-motion", action="store_true",
                   help="禁用回中减速功能")
    p.add_argument("--repeat", type=int, default=2,
                   help="每个动作重复次数 (默认 2 遍)")
    p.add_argument("--target-radius", type=float, default=0.12,
                   help="目标到达判定半径")
    
    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path, default=_ROOT / "serial_return.json")
    
    p.add_argument("--output-dir", type=str, default="outputs/8dir_comprehensive_test")
    
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("8 方向综合运动测试")
    print("=" * 70)
    print(f"串口: {args.serial_port}")
    print(f"步长: {args.step_rad} rad / {args.norm_step} (归一化)")
    print(f"测试速度: {args.velocity}")
    print(f"测试模式: {len(TEST_PATTERNS)} 个")
    print()
    print("测试流程:")
    print("  1. 同步归中 (home速度)")
    print("  2. 测试开始 [")
    print("       - 水平运动 -> 归中(速度100)")
    print("       - 垂直运动 -> 归中(速度100)")
    print("       - 斜角运动 -> 归中(速度100)")
    print("       - 正方形运动")
    print("     ] 测试结束")
    print("  3. 同步归位 (home速度)")
    print()
    
    # ========== 测试开始前: 同步归中 (home速度) ==========
    # 注意: 必须在 SerialArmEnvV2 之前调用, 因为外部脚本需要独占串口
    print("\n" + "=" * 60)
    print("测试开始前 - 同步归中 (home速度)")
    print("=" * 60)
    go_to_home(args.serial_port, args.home_json, duration=2.0)
    
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
    
    controller = SmoothMotionController(
        serial_env=serial_env,
        step_rad=args.step_rad,
        velocity=args.velocity,
        norm_step=args.norm_step,
        use_soft_motion=not args.no_soft_motion,
    )
    
    # 存储所有结果
    all_results = {}
    
    try:
        # ========== 测试开始 [ 所有动作使用速度140 ] ==========
        print("\n" + "=" * 60)
        print(f"测试开始 [ 所有动作使用速度 {args.velocity} ]")
        print("=" * 60)
        
        pattern_names = list(TEST_PATTERNS.keys())
        
        for i, name in enumerate(pattern_names):
            pattern = TEST_PATTERNS[name]
            
            # 运行测试（每个动作做两遍）
            results = run_single_pattern(
                controller=controller,
                pattern=pattern,
                target_radius=args.target_radius,
                repeat=args.repeat,
            )
            
            all_results[name] = results
            
            # 保存单个模式结果
            result_file = output_dir / f"{name}.json"
            with open(result_file, "w") as f:
                save_data = {
                    "name": results["name"],
                    "description": results["description"],
                    "targets": [list(t) for t in results["targets"]],
                    "per_target": [{
                        "target": list(pt["target"]),
                        "steps": int(pt["steps"]),
                        "reached": bool(pt["reached"]),
                        "final_dist": float(pt["final_dist"]),
                    } for pt in results["per_target"]],
                    "total_steps": int(results["total_steps"]),
                    "reached": int(results["reached"]),
                    "trajectory": [list(t) for t in results["trajectory"]],
                    "timestamps": results["timestamps"],
                }
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            
            # 绘制单个模式图
            plot_single_pattern(results, output_dir / f"{name}.png")
            print(f"  图像已保存: {output_dir / f'{name}.png'}")
            
            # 测试之间归中 - 使用测试速度100（除了最后一个）
            if i < len(pattern_names) - 1:
                controller.return_to_center()
        
        # ========== 测试结束 ==========
        print("\n" + "=" * 60)
        print("测试结束")
        print("=" * 60)
        
        # ========== 绘制综合图 ==========
        plot_all_patterns(all_results, output_dir / "all_pos_vs_time.png")
        plot_all_trajectories(all_results, output_dir / "all_trajectories.png")
        
        # 保存汇总结果
        summary = {
            "total_patterns": len(all_results),
            "patterns": list(all_results.keys()),
            "total_steps": sum(r["total_steps"] for r in all_results.values()),
            "total_targets_reached": sum(r["reached"] for r in all_results.values()),
            "total_targets": sum(len(r["targets"]) for r in all_results.values()),
        }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "=" * 70)
        print("测试完成!")
        print(f"总测试模式: {summary['total_patterns']}")
        print(f"总步数: {summary['total_steps']}")
        print(f"目标到达率: {summary['total_targets_reached']}/{summary['total_targets']}")
        print(f"结果保存: {output_dir}")
        print("=" * 70)
    
    finally:
        # 关闭环境
        serial_env.close()
        
        # 等待串口释放 (缩短时间防止漂移)
        time.sleep(0.3)
        
        # ========== 测试结束后: 先回中位再归位 ==========
        print("\n" + "=" * 60)
        print("测试结束后 - 先回中位")
        print("=" * 60)
        go_to_home(args.serial_port, args.home_json, duration=2.0)
        
        print("\n" + "=" * 60)
        print("测试结束后 - 同步归位")
        print("=" * 60)
        go_to_return(args.serial_port, args.return_json, duration=2.0)


if __name__ == "__main__":
    main()
