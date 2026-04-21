#!/usr/bin/env python3
"""
8 方向平滑运动测试

功能：
1. 匀速连续运动（无步间延迟）
2. 单独动作测试模式
3. 开始前回中位，结束后回归位

用法：
    # 测试单个动作
    python scripts/test_8dir_smooth.py --mode single --action up
    python scripts/test_8dir_smooth.py --mode single --action up_right
    
    # 测试序列
    python scripts/test_8dir_smooth.py --mode sequence --test-pattern diagonal
    
    # 所有测试
    python scripts/test_8dir_smooth.py --mode sequence --test-pattern all

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
# 匀速运动控制器
# ============================================================================

class SmoothMotionController:
    """
    匀速运动控制器
    
    关键特性：
    1. 无步间延迟 - 连续发送目标位置
    2. 对角线同步运动 - 两个关节同时移动
    3. 速度由伺服内部控制，保持匀速
    """
    
    def __init__(
        self,
        serial_env: SerialArmEnvV2,
        step_rad: float = 0.15,  # 增大步长：0.08 -> 0.15
        velocity: int = 150,
        norm_step: float = 0.15,  # 归一化步长：0.1 -> 0.15
    ):
        self.serial_env = serial_env
        self.step_rad = step_rad
        self.velocity = velocity
        self.norm_step = norm_step
        
        # 归一化位置跟踪
        self._y = 0.0
        self._z = 0.0
        
        # 轨迹记录
        self.trajectory: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.timestamps: List[float] = [0.0]
        self._start_time = time.time()
    
    def reset(self):
        """重置控制器状态"""
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self._start_time = time.time()
    
    def execute_smooth_action(self, action: int) -> Tuple[float, float]:
        """
        执行单个动作 - 匀速运动
        
        关键：同时写入两个关节目标位置，无延迟
        """
        if action not in ACTION_VECTORS:
            return (self._y, self._z)
        
        # 获取动作向量
        dy, dz = ACTION_VECTORS[action]
        
        # 计算归一化位置变化
        self._y = np.clip(self._y + dy * self.norm_step, -1.0, 1.0)
        self._z = np.clip(self._z + dz * self.norm_step, -1.0, 1.0)
        
        # 获取关节 ID
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        # 读取当前位置
        cur_lr = self.serial_env._read_ticks(id_lr)
        cur_ud = self.serial_env._read_ticks(id_ud)
        
        if cur_lr is None or cur_ud is None:
            print("[警告] 无法读取关节位置")
            return (self._y, self._z)
        
        # 计算目标位置（弧度 -> ticks）
        d_lr_rad = -dy * self.step_rad  # 负号因为方向映射
        d_ud_rad = dz * self.step_rad
        
        d_lr_ticks = So101Bus.rad_to_ticks(d_lr_rad)
        d_ud_ticks = So101Bus.rad_to_ticks(d_ud_rad)
        
        target_lr = cur_lr + d_lr_ticks
        target_ud = cur_ud + d_ud_ticks
        
        # 应用限位
        lim_lr = self.serial_env._joint_limits.get(id_lr)
        lim_ud = self.serial_env._joint_limits.get(id_ud)
        
        if lim_lr:
            target_lr = lim_lr.clamp(target_lr)
        if lim_ud:
            target_ud = lim_ud.clamp(target_ud)
        
        # ★★★ 关键：同时写入两个关节目标 ★★★
        # 不设置 time_ms，让伺服使用初始化时设置的速度匀速运动
        try:
            bus = self.serial_env._bus
            bus.write_position(id_lr, int(target_lr))
            bus.write_position(id_ud, int(target_ud))
        except Exception as e:
            print(f"[警告] 写入位置失败: {e}")
        
        # 记录轨迹和时间戳
        self.trajectory.append((self._y, self._z))
        self.timestamps.append(time.time() - self._start_time)
        
        return (self._y, self._z)
    
    def wait_motion_complete(self, timeout: float = 2.0) -> bool:
        """
        等待运动完成（位置稳定）
        """
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        
        start = time.time()
        last_lr, last_ud = None, None
        stable_count = 0
        
        while time.time() - start < timeout:
            cur_lr = self.serial_env._read_ticks(id_lr)
            cur_ud = self.serial_env._read_ticks(id_ud)
            
            if last_lr is not None and last_ud is not None:
                if abs(cur_lr - last_lr) < 5 and abs(cur_ud - last_ud) < 5:
                    stable_count += 1
                    if stable_count >= 3:
                        return True
                else:
                    stable_count = 0
            
            last_lr, last_ud = cur_lr, cur_ud
            time.sleep(0.05)
        
        return False


# ============================================================================
# 单动作测试
# ============================================================================

def test_single_action(
    serial_port: str,
    action_name: str,
    steps: int = 5,
    velocity: int = 150,
    step_rad: float = 0.15,
    norm_step: float = 0.15,
) -> Dict:
    """
    测试单个动作
    
    Args:
        action_name: 动作名称 (left, right, up, down, up_left, up_right, down_left, down_right)
        steps: 执行步数
    """
    if action_name not in NAME_TO_ACTION:
        print(f"[错误] 未知动作: {action_name}")
        print(f"可用动作: {list(NAME_TO_ACTION.keys())}")
        return None
    
    action = NAME_TO_ACTION[action_name]
    
    print("=" * 60)
    print(f"单动作测试: {action_name} (ID={action})")
    print(f"步数: {steps}, 速度: {velocity}, 步长: {step_rad} rad")
    print("=" * 60)
    
    # 初始化环境
    cfg = SerialConfigV2(
        port=serial_port,
        move_velocity=velocity,
        action_delay_ms=0,  # 无延迟
        joint_step_rad=step_rad,
    )
    
    try:
        serial_env = SerialArmEnvV2(cfg)
    except Exception as e:
        print(f"[错误] 无法初始化串口: {e}")
        return None
    
    # 创建控制器
    controller = SmoothMotionController(
        serial_env=serial_env,
        step_rad=step_rad,
        velocity=velocity,
        norm_step=norm_step,
    )
    
    # 执行动作
    print(f"\n执行 {steps} 步 {action_name} (步长={step_rad} rad)...")
    
    for i in range(steps):
        pos = controller.execute_smooth_action(action)
        print(f"  Step {i+1}: pos=({pos[0]:.3f}, {pos[1]:.3f})")
        
        # 短暂等待让伺服到达目标
        time.sleep(0.1)
    
    # 等待运动完成
    print("等待运动完成...")
    controller.wait_motion_complete(timeout=1.0)
    
    # 返回结果
    result = {
        "action": action_name,
        "steps": steps,
        "trajectory": controller.trajectory,
        "timestamps": controller.timestamps,
    }
    
    # 关闭环境
    serial_env.close()
    
    return result


# ============================================================================
# 序列测试
# ============================================================================

def test_sequence(
    serial_port: str,
    targets: List[Tuple[float, float]],
    target_radius: float = 0.15,
    max_steps: int = 30,
    velocity: int = 150,
    step_rad: float = 0.15,
    norm_step: float = 0.15,
) -> Dict:
    """
    测试目标序列 - 匀速运动模式
    """
    print("=" * 60)
    print("序列测试 - 匀速运动模式")
    print(f"目标点: {targets}")
    print(f"到达半径: {target_radius}")
    print(f"速度: {velocity}")
    print("=" * 60)
    
    # 初始化环境
    cfg = SerialConfigV2(
        port=serial_port,
        move_velocity=velocity,
        action_delay_ms=0,  # 无延迟
        joint_step_rad=step_rad,
    )
    
    try:
        serial_env = SerialArmEnvV2(cfg)
    except Exception as e:
        print(f"[错误] 无法初始化串口: {e}")
        return None
    
    # 创建控制器
    controller = SmoothMotionController(
        serial_env=serial_env,
        step_rad=step_rad,
        velocity=velocity,
        norm_step=norm_step,
    )
    
    # 结果记录
    results = {
        "targets": targets,
        "target_radius": target_radius,
        "per_target": [],
        "total_steps": 0,
        "reached": 0,
    }
    
    # 执行序列
    for i, (ty, tz) in enumerate(targets):
        print(f"\n[目标 {i+1}/{len(targets)}] -> ({ty:.2f}, {tz:.2f})")
        
        steps = 0
        reached = False
        
        while steps < max_steps:
            # 计算到目标的距离
            dist = np.sqrt((controller._y - ty)**2 + (controller._z - tz)**2)
            
            if dist < target_radius:
                reached = True
                print(f"  ✓ 到达! 步数={steps}, 距离={dist:.3f}")
                break
            
            # 选择最优动作
            best_action = None
            best_dot = -float('inf')
            
            dir_to_target = np.array([ty - controller._y, tz - controller._z])
            dir_to_target = dir_to_target / (np.linalg.norm(dir_to_target) + 1e-8)
            
            for action, vec in ACTION_VECTORS.items():
                dot = np.dot(dir_to_target, vec)
                if dot > best_dot:
                    best_dot = dot
                    best_action = action
            
            # 执行动作
            controller.execute_smooth_action(best_action)
            steps += 1
            
            # 匀速运动：极短等待（只是为了让伺服处理命令）
            time.sleep(0.05)
            
            if steps % 5 == 0:
                print(f"  Step {steps}: {ACTION_NAMES[best_action]}, "
                      f"pos=({controller._y:.2f}, {controller._z:.2f}), dist={dist:.3f}")
        
        # 等待运动完成
        controller.wait_motion_complete(timeout=0.5)
        
        results["per_target"].append({
            "target": (ty, tz),
            "steps": steps,
            "reached": reached,
            "final_dist": dist,
        })
        results["total_steps"] += steps
        if reached:
            results["reached"] += 1
    
    results["trajectory"] = controller.trajectory
    results["timestamps"] = controller.timestamps
    
    # 关闭环境
    serial_env.close()
    
    return results


# ============================================================================
# 归位函数
# ============================================================================

def go_to_home(serial_port: str, home_json: Path, wait_time: float = 2.0):
    """
    回到中位
    
    Args:
        wait_time: 回中后额外等待时间(秒)，确保所有关节到位
    """
    if not home_json.exists():
        print(f"[警告] 中位文件不存在: {home_json}")
        return
    
    print("\n" + "=" * 60)
    print("回到中位...")
    print("=" * 60)
    
    # 使用较长的 time-ms 确保运动完成
    cmd = f"python {_HERE}/serial_go_home.py --port {serial_port} --home-json {home_json} --time-ms 1000 --wait 1.5"
    os.system(cmd)
    
    # 额外等待确保所有关节完全到位
    print(f"等待关节到位 ({wait_time}s)...")
    time.sleep(wait_time)


def go_to_return(serial_port: str, return_json: Path, wait_time: float = 2.0):
    """
    回到归位
    
    Args:
        wait_time: 归位后额外等待时间(秒)，确保所有关节到位
    """
    if not return_json.exists():
        print(f"[警告] 归位文件不存在: {return_json}")
        return
    
    print("\n" + "=" * 60)
    print("回到归位...")
    print("=" * 60)
    
    # 使用较长的 time-ms 确保运动完成
    cmd = f"python {_HERE}/serial_go_return.py --port {serial_port} --return-json {return_json} --time-ms 1000 --wait 1.5"
    os.system(cmd)
    
    # 额外等待确保所有关节完全到位
    print(f"等待关节到位 ({wait_time}s)...")
    time.sleep(wait_time)


# ============================================================================
# 可视化
# ============================================================================

def visualize_results(results: Dict, output_path: Path, title: str = "8-Dir Smooth Test"):
    """生成可视化图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 2D 轨迹
    ax1 = axes[0]
    traj = np.array(results["trajectory"])
    ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2, label='Trajectory')
    ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=12, label='Start')
    
    if "targets" in results:
        for t in results["targets"]:
            circle = plt.Circle(t, results.get("target_radius", 0.15),
                              fill=False, color='r', linestyle='--', linewidth=2)
            ax1.add_patch(circle)
            ax1.plot(t[0], t[1], 'r*', markersize=15)
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_xlabel('Y Position (normalized)')
    ax1.set_ylabel('Z Position (normalized)')
    ax1.set_title('2D Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Position vs Time
    ax2 = axes[1]
    timestamps = results.get("timestamps", list(range(len(traj))))
    ax2.plot(timestamps, traj[:, 0], 'b-', linewidth=2, label='Y (left-right)')
    ax2.plot(timestamps, traj[:, 1], 'g-', linewidth=2, label='Z (up-down)')
    
    if "targets" in results:
        for i, t in enumerate(results["targets"]):
            ax2.axhline(y=t[0], color='b', linestyle='--', alpha=0.5)
            ax2.axhline(y=t[1], color='g', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Position')
    ax2.set_title('Position vs Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"图像已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="8 方向平滑运动测试")
    
    # 模式
    p.add_argument("--mode", type=str, default="single",
                   choices=["single", "sequence"],
                   help="测试模式: single=单动作, sequence=序列")
    
    # 单动作参数
    p.add_argument("--action", type=str, default="up",
                   choices=list(NAME_TO_ACTION.keys()),
                   help="单动作测试的动作名称")
    p.add_argument("--steps", type=int, default=5,
                   help="单动作测试的步数")
    
    # 序列参数
    p.add_argument("--test-pattern", type=str, default="diagonal",
                   choices=["diagonal", "square", "horizontal", "all"],
                   help="序列测试的目标模式")
    
    # 运动参数
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--velocity", type=int, default=150,
                   help="伺服运动速度")
    p.add_argument("--step-rad", type=float, default=0.15,
                   help="每步弧度 (增大=运动幅度更大)")
    p.add_argument("--norm-step", type=float, default=0.15,
                   help="归一化步长 (增大=运动幅度更大)")
    p.add_argument("--target-radius", type=float, default=0.15)
    
    # 归位参数
    p.add_argument("--pre-home", action="store_true",
                   help="测试前回到中位")
    p.add_argument("--post-return", action="store_true",
                   help="测试后回到归位")
    p.add_argument("--home-json", type=Path, 
                   default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path,
                   default=_ROOT / "serial_return.json")
    
    # 输出
    p.add_argument("--output-dir", type=str, 
                   default="outputs/8dir_smooth_test")
    
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== 测试前：回到中位 ==========
    if args.pre_home:
        go_to_home(args.serial_port, args.home_json)
    
    # ========== 执行测试 ==========
    try:
        if args.mode == "single":
            # 单动作测试
            result = test_single_action(
                serial_port=args.serial_port,
                action_name=args.action,
                steps=args.steps,
                velocity=args.velocity,
                step_rad=args.step_rad,
                norm_step=args.norm_step,
            )
            
            if result:
                # 保存结果
                with open(output_dir / f"single_{args.action}.json", "w") as f:
                    json.dump({
                        "action": result["action"],
                        "steps": result["steps"],
                        "trajectory": [list(t) for t in result["trajectory"]],
                        "timestamps": result["timestamps"],
                    }, f, indent=2)
                
                # 可视化
                visualize_results(
                    result, 
                    output_dir / f"single_{args.action}.png",
                    title=f"Single Action Test: {args.action}"
                )
        
        else:  # sequence
            # 定义测试模式
            patterns = {
                "diagonal": [(0.5, 0.5), (-0.5, -0.5), (0.0, 0.0)],
                "square": [(0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)],
                "horizontal": [(0.6, 0.0), (-0.6, 0.0), (0.0, 0.0)],
            }
            
            if args.test_pattern == "all":
                for name, targets in patterns.items():
                    print(f"\n{'=' * 60}")
                    print(f"测试模式: {name}")
                    print(f"{'=' * 60}")
                    
                    result = test_sequence(
                        serial_port=args.serial_port,
                        targets=targets,
                        target_radius=args.target_radius,
                        velocity=args.velocity,
                        step_rad=args.step_rad,
                        norm_step=args.norm_step,
                    )
                    
                    if result:
                        # 保存
                        with open(output_dir / f"sequence_{name}.json", "w") as f:
                            def to_serializable(obj):
                                if isinstance(obj, (np.bool_, np.integer)):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, tuple):
                                    return list(obj)
                                return obj
                            
                            save_data = {
                                "targets": [list(t) for t in result["targets"]],
                                "target_radius": result["target_radius"],
                                "per_target": [{
                                    "target": list(pt["target"]),
                                    "steps": int(pt["steps"]),
                                    "reached": bool(pt["reached"]),
                                    "final_dist": float(pt["final_dist"]),
                                } for pt in result["per_target"]],
                                "total_steps": int(result["total_steps"]),
                                "reached": int(result["reached"]),
                                "trajectory": [list(t) for t in result["trajectory"]],
                                "timestamps": result["timestamps"],
                            }
                            json.dump(save_data, f, indent=2)
                        
                        visualize_results(
                            result,
                            output_dir / f"sequence_{name}.png",
                            title=f"Sequence Test: {name}"
                        )
                    
                    # 测试之间回中
                    go_to_home(args.serial_port, args.home_json)
            
            else:
                targets = patterns[args.test_pattern]
                result = test_sequence(
                    serial_port=args.serial_port,
                    targets=targets,
                    target_radius=args.target_radius,
                    velocity=args.velocity,
                    step_rad=args.step_rad,
                    norm_step=args.norm_step,
                )
                
                if result:
                    with open(output_dir / f"sequence_{args.test_pattern}.json", "w") as f:
                        save_data = {
                            "targets": [list(t) for t in result["targets"]],
                            "target_radius": result["target_radius"],
                            "per_target": [{
                                "target": list(pt["target"]),
                                "steps": int(pt["steps"]),
                                "reached": bool(pt["reached"]),
                                "final_dist": float(pt["final_dist"]),
                            } for pt in result["per_target"]],
                            "total_steps": int(result["total_steps"]),
                            "reached": int(result["reached"]),
                            "trajectory": [list(t) for t in result["trajectory"]],
                            "timestamps": result["timestamps"],
                        }
                        json.dump(save_data, f, indent=2)
                    
                    visualize_results(
                        result,
                        output_dir / f"sequence_{args.test_pattern}.png",
                        title=f"Sequence Test: {args.test_pattern}"
                    )
        
        print(f"\n结果已保存到: {output_dir}")
    
    finally:
        # ========== 测试后：回到归位 ==========
        if args.post_return:
            go_to_return(args.serial_port, args.return_json)


if __name__ == "__main__":
    main()
