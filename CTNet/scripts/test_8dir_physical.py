#!/usr/bin/env python3
"""
物理机械臂 8 方向运动测试

使用爪子尖端 (gripper tip) 作为目标点，测试 8 方向运动的平滑性。

用法：
    python scripts/test_8dir_physical.py --serial-port /dev/ttyACM1
    
创建时间: 2026-02-25
"""

from __future__ import annotations

import argparse
import json
import math
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
# SO-101 机械臂正向运动学 (Forward Kinematics)
# ============================================================================

@dataclass
class ArmDimensions:
    """SO-101 机械臂连杆尺寸 (单位: 米)"""
    # 基座到 shoulder_pan 轴
    base_height: float = 0.05
    
    # shoulder_pan 到 shoulder_lift
    shoulder_offset: float = 0.03
    
    # 上臂长度 (shoulder_lift 到 elbow_flex)
    upper_arm: float = 0.10
    
    # 前臂长度 (elbow_flex 到 wrist_flex)
    forearm: float = 0.10
    
    # 腕到爪子尖端
    wrist_to_gripper: float = 0.08
    
    # 总臂展 (全部伸直时)
    @property
    def total_reach(self) -> float:
        return self.upper_arm + self.forearm + self.wrist_to_gripper


class GripperTipTracker:
    """
    爪子尖端位置跟踪器
    
    使用简化的 2D 正向运动学，基于 shoulder_pan 和 wrist_flex 两个关节。
    假设其他关节固定在中间位置。
    """
    
    def __init__(self, arm_dims: ArmDimensions = None):
        self.dims = arm_dims or ArmDimensions()
        
        # 基准臂长 (当 wrist_flex 在中间位置时)
        # 简化模型：假设手臂水平伸出
        self.base_arm_length = self.dims.upper_arm + self.dims.forearm + self.dims.wrist_to_gripper
        
        # 初始关节角度 (归一化: -1 到 1)
        self._pan_normalized = 0.0   # shoulder_pan: -1=左, 0=中, 1=右
        self._flex_normalized = 0.0  # wrist_flex: -1=下, 0=中, 1=上
        
        # 工作空间范围
        self.workspace_y = (-0.3, 0.3)  # 左右 (米)
        self.workspace_z = (-0.15, 0.15)  # 上下 (米)
    
    def update_from_ticks(self, pan_ticks: int, flex_ticks: int,
                          pan_limits: Tuple[int, int],
                          flex_limits: Tuple[int, int]):
        """
        从关节 ticks 更新爪子尖端位置
        
        Args:
            pan_ticks: shoulder_pan 当前 ticks
            flex_ticks: wrist_flex 当前 ticks
            pan_limits: (min_ticks, max_ticks) for shoulder_pan
            flex_limits: (min_ticks, max_ticks) for wrist_flex
        """
        # 归一化到 [-1, 1]
        pan_range = pan_limits[1] - pan_limits[0]
        pan_center = (pan_limits[0] + pan_limits[1]) / 2
        self._pan_normalized = (pan_ticks - pan_center) / (pan_range / 2)
        
        flex_range = flex_limits[1] - flex_limits[0]
        flex_center = (flex_limits[0] + flex_limits[1]) / 2
        self._flex_normalized = (flex_ticks - flex_center) / (flex_range / 2)
    
    def get_gripper_tip_position(self) -> Tuple[float, float, float]:
        """
        计算爪子尖端在笛卡尔空间中的位置
        
        Returns:
            (x, y, z): 爪子尖端位置 (米)
            x: 前后 (固定)
            y: 左右 (shoulder_pan 控制)
            z: 上下 (wrist_flex 控制)
        """
        # 简化模型:
        # - Y 位置由 shoulder_pan 控制 (左右摆动)
        # - Z 位置由 wrist_flex 控制 (上下摆动)
        # - X 位置固定 (手臂伸出方向)
        
        # Y: 左右位置
        # 假设 shoulder_pan 范围对应 ±30 度
        pan_angle_rad = self._pan_normalized * math.radians(30)
        y = self.base_arm_length * math.sin(pan_angle_rad)
        
        # Z: 上下位置
        # 假设 wrist_flex 范围对应 ±20 度
        flex_angle_rad = self._flex_normalized * math.radians(20)
        z = self.base_arm_length * math.sin(flex_angle_rad) + self.dims.base_height
        
        # X: 固定前伸
        x = self.base_arm_length * math.cos(pan_angle_rad) * math.cos(flex_angle_rad)
        
        return (x, y, z)
    
    def get_normalized_position(self) -> Tuple[float, float]:
        """
        获取归一化位置 (用于 RL 状态)
        
        Returns:
            (y_norm, z_norm): 归一化到 [-1, 1] 的 Y, Z 位置
        """
        return (self._pan_normalized, self._flex_normalized)


# ============================================================================
# 8 方向控制器
# ============================================================================

class EightDirectionController:
    """8 方向物理控制器"""
    
    # 8 方向动作向量 (Y, Z)
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
    
    ACTION_NAMES = {
        0: "left", 1: "right", 2: "up", 3: "down",
        4: "up_left", 5: "up_right", 6: "down_left", 7: "down_right",
    }
    
    def __init__(
        self,
        serial_env: SerialArmEnvV2,
        step_size: float = 0.1,
        adaptive_step: bool = True,
        max_step_size: float = 0.15,
        min_step_size: float = 0.05,
    ):
        self.serial_env = serial_env
        self.step_size = step_size
        self.adaptive_step = adaptive_step
        self.max_step_size = max_step_size
        self.min_step_size = min_step_size
        
        # 爪子尖端跟踪
        self.tracker = GripperTipTracker()
        
        # 当前归一化位置
        self._y = 0.0
        self._z = 0.0
        
        # 轨迹记录
        self.trajectory: List[Tuple[float, float]] = []
        self.gripper_trajectory: List[Tuple[float, float, float]] = []
    
    def get_optimal_action(self, target_y: float, target_z: float) -> int:
        """选择最优方向"""
        dy = target_y - self._y
        dz = target_z - self._z
        
        if abs(dy) < 0.01 and abs(dz) < 0.01:
            return 0
        
        target_vec = np.array([dy, dz])
        target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-8)
        
        best_action = 0
        best_dot = -2.0
        
        for action, (vy, vz) in self.ACTION_VECTORS.items():
            action_vec = np.array([vy, vz])
            dot = np.dot(target_vec, action_vec)
            if dot > best_dot:
                best_dot = dot
                best_action = action
        
        return best_action
    
    def step(self, action: int) -> Tuple[float, float]:
        """执行一步动作 - 真正的对角线同步运动"""
        if action not in self.ACTION_VECTORS:
            return (self._y, self._z)
        
        # 计算步长
        if self.adaptive_step:
            dist = np.sqrt((self._y)**2 + (self._z)**2)
            current_step = min(self.max_step_size, max(self.min_step_size, dist * 0.3))
        else:
            current_step = self.step_size
        
        # 获取动作向量
        dy, dz = self.ACTION_VECTORS[action]
        dy *= current_step
        dz *= current_step
        
        # 更新虚拟位置
        self._y = np.clip(self._y + dy, -1.0, 1.0)
        self._z = np.clip(self._z + dz, -1.0, 1.0)
        
        # 发送到物理机械臂 - 使用同步对角线运动
        self._execute_diagonal_move(action)
        
        # 记录轨迹
        self.trajectory.append((self._y, self._z))
        
        # 读取实际爪子尖端位置
        try:
            gripper_pos = self._read_gripper_position()
            self.gripper_trajectory.append(gripper_pos)
        except:
            pass
        
        return (self._y, self._z)
    
    def _execute_diagonal_move(self, action: int):
        """
        执行真正的对角线同步运动
        
        关键：同时写入两个关节的目标位置，让它们同步运动
        """
        # 获取动作向量
        dy, dz = self.ACTION_VECTORS[action]
        
        # 计算两个关节的步长
        step_rad = self.serial_env.cfg.joint_step_rad
        
        # 获取关节 ID
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        # 读取当前位置
        cur_lr = self.serial_env._read_ticks(id_lr)
        cur_ud = self.serial_env._read_ticks(id_ud)
        
        if cur_lr is None or cur_ud is None:
            print("[警告] 无法读取关节位置")
            return
        
        # 计算目标位置
        d_lr_rad = -dy * step_rad  # 负号因为方向映射
        d_ud_rad = dz * step_rad
        
        d_lr_ticks = So101Bus.rad_to_ticks(d_lr_rad)
        d_ud_ticks = So101Bus.rad_to_ticks(d_ud_rad)
        
        target_lr = cur_lr + d_lr_ticks
        target_ud = cur_ud + d_ud_ticks
        
        # 应用软限位
        lim_lr = self.serial_env._joint_limits.get(id_lr)
        lim_ud = self.serial_env._joint_limits.get(id_ud)
        
        if lim_lr:
            target_lr = lim_lr.clamp(target_lr)
        if lim_ud:
            target_ud = lim_ud.clamp(target_ud)
        
        # ★★★ 关键：同时写入两个关节的目标位置 ★★★
        # 这样两个关节会同步运动，形成真正的对角线轨迹
        try:
            self.serial_env._smooth_move(id_lr, target_lr)
            self.serial_env._smooth_move(id_ud, target_ud)
        except Exception as e:
            print(f"[警告] 关节运动失败: {e}")
    
    def _read_gripper_position(self) -> Tuple[float, float, float]:
        """读取爪子尖端位置"""
        # 从 serial_env 读取关节位置
        pan_ticks = self.serial_env._read_ticks(self.serial_env._id_lr)
        flex_ticks = self.serial_env._read_ticks(self.serial_env._id_ud)
        
        if pan_ticks is None or flex_ticks is None:
            return (0.0, 0.0, 0.0)
        
        # 获取限位
        pan_lim = self.serial_env._joint_limits.get(self.serial_env._id_lr)
        flex_lim = self.serial_env._joint_limits.get(self.serial_env._id_ud)
        
        if pan_lim and flex_lim:
            self.tracker.update_from_ticks(
                pan_ticks, flex_ticks,
                (pan_lim.min_ticks, pan_lim.max_ticks),
                (flex_lim.min_ticks, flex_lim.max_ticks),
            )
        
        return self.tracker.get_gripper_tip_position()
    
    def reset(self):
        """重置"""
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.gripper_trajectory = []
        self.serial_env.reset(options={"recenter": True})
        
        # 读取初始爪子位置
        try:
            gripper_pos = self._read_gripper_position()
            self.gripper_trajectory.append(gripper_pos)
        except:
            pass
    
    def distance_to_target(self, target_y: float, target_z: float) -> float:
        """计算到目标的距离"""
        return np.sqrt((self._y - target_y)**2 + (self._z - target_z)**2)


# ============================================================================
# 测试函数
# ============================================================================

def run_8dir_test(
    serial_port: str,
    targets: List[Tuple[float, float]],
    target_radius: float = 0.15,
    max_steps_per_target: int = 30,
    action_delay: float = 0.15,  # 减小延迟实现更平滑的运动
    move_velocity: int = 150,   # 提高速度实现匀速运动
    output_dir: Path = None,
):
    """
    运行 8 方向物理测试
    
    参数调优指南：
    - action_delay: 每步之间的等待时间，越小运动越连续平滑
    - move_velocity: 伺服速度，越高运动越快
    
    平滑运动 = 高速度 + 短延迟
    """
    
    print("=" * 60)
    print("物理机械臂 8 方向运动测试")
    print("=" * 60)
    print(f"串口: {serial_port}")
    print(f"目标点: {targets}")
    print(f"到达半径: {target_radius}")
    print(f"运动速度: {move_velocity}")
    print()
    
    # 初始化串口环境
    cfg = SerialConfigV2(
        port=serial_port,
        move_velocity=move_velocity,
        action_delay_ms=int(action_delay * 1000),
        joint_step_rad=0.08,  # 适中的步长
    )
    
    try:
        serial_env = SerialArmEnvV2(cfg)
    except Exception as e:
        print(f"串口初始化失败: {e}")
        print("请检查:")
        print("  1. 机械臂是否已连接")
        print("  2. 串口权限: sudo chmod 666 /dev/ttyACM1")
        return None
    
    # 初始化控制器
    controller = EightDirectionController(
        serial_env=serial_env,
        adaptive_step=True,
        max_step_size=0.15,
        min_step_size=0.05,
    )
    
    # 重置到中心
    print("重置机械臂到中心位置...")
    controller.reset()
    time.sleep(1.0)
    
    # 运行测试
    results = {
        "targets": targets,
        "target_radius": target_radius,
        "per_target": [],
        "total_steps": 0,
        "reached": 0,
    }
    
    for i, (ty, tz) in enumerate(targets):
        print(f"\n[目标 {i+1}/{len(targets)}] -> ({ty:.2f}, {tz:.2f})")
        
        steps = 0
        start_pos = (controller._y, controller._z)
        
        while steps < max_steps_per_target:
            dist = controller.distance_to_target(ty, tz)
            
            if dist < target_radius:
                print(f"  ✓ 到达! 步数={steps}, 距离={dist:.3f}")
                results["reached"] += 1
                break
            
            # 选择最优动作
            action = controller.get_optimal_action(ty, tz)
            action_name = controller.ACTION_NAMES[action]
            
            # 执行动作
            controller.step(action)
            steps += 1
            
            # 等待运动完成
            time.sleep(action_delay)
            
            if steps % 5 == 0:
                print(f"  Step {steps}: {action_name}, pos=({controller._y:.2f}, {controller._z:.2f}), dist={dist:.3f}")
        
        results["per_target"].append({
            "target": (ty, tz),
            "start": start_pos,
            "end": (controller._y, controller._z),
            "steps": steps,
            "reached": dist < target_radius,
            "final_distance": dist,
        })
        results["total_steps"] += steps
    
    # 保存轨迹
    results["trajectory"] = controller.trajectory
    results["gripper_trajectory"] = controller.gripper_trajectory
    
    # 回到中心位置
    print("\n回到中心位置...")
    controller.serial_env.reset(options={"recenter": True})
    time.sleep(1.5)
    
    # 关闭环境
    print("关闭机械臂...")
    serial_env.close()
    
    # 可视化
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 绘制轨迹图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 2D 轨迹
        ax = axes[0]
        traj = np.array(controller.trajectory)
        ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
        ax.plot(0, 0, 'go', markersize=12, label='Start')
        for ty, tz in targets:
            ax.plot(ty, tz, 'r*', markersize=15)
            circle = plt.Circle((ty, tz), target_radius, fill=False, color='red', linestyle='--')
            ax.add_patch(circle)
        ax.set_xlabel('Y Position (normalized)')
        ax.set_ylabel('Z Position (normalized)')
        ax.set_title('8-Direction Physical Control - 2D Trajectory')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Position vs Time
        ax = axes[1]
        steps_arr = list(range(len(controller.trajectory)))
        ys = [p[0] for p in controller.trajectory]
        zs = [p[1] for p in controller.trajectory]
        ax.plot(steps_arr, ys, 'b-', linewidth=2, label='Y (left-right)')
        ax.plot(steps_arr, zs, 'g-', linewidth=2, label='Z (up-down)')
        for ty, tz in targets:
            ax.axhline(y=ty, color='blue', linestyle='--', alpha=0.5)
            ax.axhline(y=tz, color='green', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Position')
        ax.set_title('Position vs Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.suptitle(f'8-Direction Physical Test\nTotal: {results["total_steps"]} steps, Reached: {results["reached"]}/{len(targets)}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "8dir_physical_test.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 保存结果
        with open(output_dir / "8dir_physical_results.json", "w") as f:
            # 转换所有数据为 JSON 可序列化格式
            def convert_to_json_serializable(obj):
                if isinstance(obj, (np.bool_, np.integer)):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(i) for i in obj]
                elif isinstance(obj, bool):
                    return obj
                return obj
            
            results_save = {
                "targets": [list(t) for t in results["targets"]],
                "target_radius": results["target_radius"],
                "per_target": convert_to_json_serializable(results["per_target"]),
                "total_steps": int(results["total_steps"]),
                "reached": int(results["reached"]),
                "trajectory": [list(t) for t in results["trajectory"]],
            }
            json.dump(results_save, f, indent=2)
        
        print(f"\n结果已保存到: {output_dir}")
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    print(f"总步数: {results['total_steps']}")
    print(f"到达率: {results['reached']}/{len(targets)} ({results['reached']/len(targets)*100:.1f}%)")
    
    return results


def parse_args():
    p = argparse.ArgumentParser(description="物理机械臂 8 方向测试")
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--target-radius", type=float, default=0.15)
    p.add_argument("--action-delay", type=float, default=0.15,
                   help="每步之间的等待时间(秒)，越小运动越平滑")
    p.add_argument("--move-velocity", type=int, default=150,
                   help="伺服运动速度，越高运动越快")
    p.add_argument("--output-dir", type=str, default="outputs/8dir_physical_test")
    p.add_argument("--test-pattern", type=str, default="diagonal",
                   choices=["diagonal", "square", "horizontal", "all"])
    return p.parse_args()


def main():
    args = parse_args()
    
    # 定义测试模式
    patterns = {
        "diagonal": [(0.5, 0.5), (-0.5, -0.5), (0.0, 0.0)],
        "square": [(0.5, 0.0), (0.0, 0.5), (-0.5, 0.0), (0.0, -0.5), (0.0, 0.0)],
        "horizontal": [(0.5, 0.0), (-0.5, 0.0), (0.0, 0.0)],
    }
    
    if args.test_pattern == "all":
        for name, targets in patterns.items():
            print(f"\n{'='*60}")
            print(f"测试模式: {name}")
            print(f"{'='*60}")
            run_8dir_test(
                serial_port=args.serial_port,
                targets=targets,
                target_radius=args.target_radius,
                action_delay=args.action_delay,
                move_velocity=args.move_velocity,
                output_dir=Path(args.output_dir) / name,
            )
    else:
        targets = patterns[args.test_pattern]
        run_8dir_test(
            serial_port=args.serial_port,
            targets=targets,
            target_radius=args.target_radius,
            action_delay=args.action_delay,
            move_velocity=args.move_velocity,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
