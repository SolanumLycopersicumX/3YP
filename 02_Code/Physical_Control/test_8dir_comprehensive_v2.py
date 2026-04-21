#!/usr/bin/env python3
"""
8 方向综合运动测试 V2

相比 V1 (test_8dir_comprehensive.py) 的改进:
  1. 更大运动幅度: 减小 norm_step (0.25→0.10), 增加 max_steps (30→50)
     使每个目标需要更多步到达, 从而让机械臂的物理运动距离增大
  2. 移除软限位: soft_limit_margin=0.0, 让机械臂能使用全部物理行程
  3. 修复归一化位置追踪 bug (v1 的 execute_action 中 _y/_z 未更新)

测试序列 (与 V1 一致):
  a. 水平运动 (左→右→中)
  b. 垂直运动 (上→下→中)
  c. 左上到右下斜角
  d. 右上到左下斜角
  e. 顺时针正方形
  f. 逆时针正方形

用法:
    python scripts/test_8dir_comprehensive_v2.py --serial-port /dev/ttyACM1

创建时间: 2026-03-04
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2
from drivers.so101_serial import So101Bus


# ============================================================================
# 8 方向动作
# ============================================================================

ACTION_NAMES = {
    0: "left", 1: "right", 2: "up", 3: "down",
    4: "up_left", 5: "up_right", 6: "down_left", 7: "down_right",
}

ACTION_VECTORS = {
    0: (-1.0, 0.0),
    1: (1.0, 0.0),
    2: (0.0, 1.0),
    3: (0.0, -1.0),
    4: (-0.707, 0.707),
    5: (0.707, 0.707),
    6: (-0.707, -0.707),
    7: (0.707, -0.707),
}


# ============================================================================
# 测试序列
# ============================================================================

@dataclass
class TestPattern:
    name: str
    description: str
    targets: List[Tuple[float, float]]


TEST_PATTERNS = {
    "a_horizontal": TestPattern(
        name="a_horizontal",
        description="Horizontal (Left→Right→Center)",
        targets=[(-1.00, 0.0), (1.00, 0.0), (0.0, 0.0)],
    ),
    "b_vertical": TestPattern(
        name="b_vertical",
        description="Vertical (Up→Down→Center)",
        targets=[(0.0, 1.00), (0.0, -1.00), (0.0, 0.0)],
    ),
    "c_diagonal_ul_dr": TestPattern(
        name="c_diagonal_ul_dr",
        description="Diagonal (UpperLeft→LowerRight)",
        targets=[(-0.85, 0.85), (0.85, -0.85), (0.0, 0.0)],
    ),
    "d_diagonal_ur_dl": TestPattern(
        name="d_diagonal_ur_dl",
        description="Diagonal (UpperRight→LowerLeft)",
        targets=[(0.85, 0.85), (-0.85, -0.85), (0.0, 0.0)],
    ),
    "e_square_cw": TestPattern(
        name="e_square_cw",
        description="Square (Clockwise)",
        targets=[(0.85, 0.0), (0.85, -0.85), (-0.85, -0.85),
                 (-0.85, 0.85), (0.85, 0.85), (0.0, 0.0)],
    ),
    "f_square_ccw": TestPattern(
        name="f_square_ccw",
        description="Square (Counter-Clockwise)",
        targets=[(0.0, 0.85), (0.85, 0.85), (0.85, -0.85),
                 (-0.85, -0.85), (-0.85, 0.85), (0.0, 0.0)],
    ),
}


# ============================================================================
# 运动控制器 V2
# ============================================================================

class SmoothMotionControllerV2:
    """
    V2 运动控制器

    改进:
    - 减小 norm_step 使每个目标需要更多步 → 更大物理运动幅度
    - 修复归一化位置追踪
    - 移除软限位 (由 SerialConfigV2 的 soft_limit_margin=0 控制)
    """

    def __init__(
        self,
        serial_env: SerialArmEnvV2,
        step_rad: float = 0.35,
        velocity: int = 140,
        norm_step: float = 0.10,
    ):
        self.serial_env = serial_env
        self.step_rad = step_rad
        self.velocity = velocity
        self.norm_step = norm_step

        self._y = 0.0
        self._z = 0.0

        self.trajectory: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.timestamps: List[float] = [0.0]
        self._start_time = time.time()

        self._enable_all_torque()
        self._set_test_velocity()

    def _enable_all_torque(self):
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
        try:
            bus = self.serial_env._bus
            id_lr = self.serial_env._id_lr
            id_ud = self.serial_env._id_ud

            v = self.velocity
            vel_bytes = bytes([v & 0xFF, (v >> 8) & 0xFF])
            time_ms = 2000
            time_bytes = bytes([time_ms & 0xFF, (time_ms >> 8) & 0xFF])

            bus.write(id_lr, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_ud, bus.GOAL_VELOCITY, vel_bytes)
            bus.write(id_lr, bus.GOAL_TIME, time_bytes)
            bus.write(id_ud, bus.GOAL_TIME, time_bytes)

            print(f"[测试速度] velocity={v}, time={time_ms}ms")
        except Exception as e:
            print(f"[警告] 设置速度失败: {e}")

    def reset(self):
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self._start_time = time.time()
        self._set_test_velocity()

    def _clamp_to_hw_limits(self, jid: int, ticks: int) -> int:
        """钳制到硬件限位，防止舵机在限位处震动"""
        lim = self.serial_env._joint_limits.get(jid)
        if lim is not None and lim.min_ticks is not None and lim.max_ticks is not None:
            return max(lim.min_ticks, min(lim.max_ticks, ticks))
        return ticks

    def execute_action(self, action: int) -> Tuple[float, float]:
        if action not in ACTION_VECTORS:
            return (self._y, self._z)

        dy, dz = ACTION_VECTORS[action]

        old_y, old_z = self._y, self._z
        self._y = np.clip(self._y + dy * self.norm_step, -1.5, 1.5)
        self._z = np.clip(self._z + dz * self.norm_step, -1.5, 1.5)

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

        target_lr = self._clamp_to_hw_limits(id_lr, cur_lr + d_lr_ticks)
        target_ud = self._clamp_to_hw_limits(id_ud, cur_ud + d_ud_ticks)

        try:
            bus = self.serial_env._bus
            bus.write_position(id_lr, int(target_lr))
            bus.write_position(id_ud, int(target_ud))
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
        max_steps: int = 50,
    ) -> Tuple[int, bool, float]:
        ty, tz = target
        steps = 0

        while steps < max_steps:
            dist = np.sqrt((self._y - ty) ** 2 + (self._z - tz) ** 2)

            if dist < target_radius:
                return steps, True, dist

            dir_to_target = np.array([ty - self._y, tz - self._z])
            dir_to_target = dir_to_target / (np.linalg.norm(dir_to_target) + 1e-8)

            best_action = 0
            best_dot = -float("inf")

            for action, vec in ACTION_VECTORS.items():
                dot = np.dot(dir_to_target, vec)
                if dot > best_dot:
                    best_dot = dot
                    best_action = action

            self.execute_action(best_action)
            steps += 1

        final_dist = np.sqrt((self._y - ty) ** 2 + (self._z - tz) ** 2)
        return steps, False, final_dist

    def return_to_center(self):
        print("\n  [归中] 使用测试速度...")

        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud

        center_lr = self.serial_env._joint_centers.get(id_lr)
        center_ud = self.serial_env._joint_centers.get(id_ud)

        if center_lr is None or center_ud is None:
            print("  [警告] 无法获取中心位置")
            return

        try:
            bus = self.serial_env._bus
            bus.write_position(id_lr, center_lr)
            bus.write_position(id_ud, center_ud)
            time.sleep(1.5)
        except Exception as e:
            print(f"  [警告] 归中失败: {e}")

        self._y = 0.0
        self._z = 0.0
        self.trajectory.append((0.0, 0.0))
        self.timestamps.append(time.time() - self._start_time)
        print("  [归中] 完成")


# ============================================================================
# 归位 (同步模式)
# ============================================================================

def go_to_home(serial_port: str, home_json: Path, duration: float = 2.0):
    if not home_json.exists():
        print(f"[警告] 中位文件不存在: {home_json}")
        return
    print("\n[同步回中位]")
    cmd = (f"python {_HERE}/serial_go_home_sync.py "
           f"--port {serial_port} --home-json {home_json} "
           f"--duration {duration} --wait 1.5")
    os.system(cmd)
    time.sleep(0.5)


def go_to_return(serial_port: str, return_json: Path, duration: float = 2.0):
    if not return_json.exists():
        print(f"[警告] 归位文件不存在: {return_json}")
        return
    print("\n[同步归位]")
    cmd = (f"python {_HERE}/serial_go_return_sync.py "
           f"--port {serial_port} --return-json {return_json} "
           f"--duration {duration} --wait 1.5")
    os.system(cmd)


# ============================================================================
# 测试
# ============================================================================

def run_single_pattern(
    controller: SmoothMotionControllerV2,
    pattern: TestPattern,
    target_radius: float = 0.12,
    max_steps: int = 50,
    repeat: int = 2,
) -> Dict:
    print(f"\n{'=' * 60}")
    print(f"测试: {pattern.description} (重复 {repeat} 次, max_steps={max_steps})")
    print(f"目标序列: {pattern.targets}")
    print(f"{'=' * 60}")

    controller.reset()

    results = {
        "name": pattern.name,
        "description": pattern.description,
        "targets": pattern.targets,
        "repeat": repeat,
        "max_steps": max_steps,
        "per_target": [],
        "total_steps": 0,
        "reached": 0,
    }

    total_targets = len(pattern.targets) * repeat
    target_count = 0

    for rep in range(repeat):
        if repeat > 1:
            print(f"  --- 第 {rep + 1}/{repeat} 遍 ---")

        for target in pattern.targets:
            target_count += 1
            print(f"  [{target_count}/{total_targets}] "
                  f"-> ({target[0]:.2f}, {target[1]:.2f})", end="")

            steps, reached, dist = controller.move_to_target(
                target, target_radius=target_radius, max_steps=max_steps,
            )

            status = "✓" if reached else "✗"
            print(f" {status} steps={steps}, dist={dist:.3f}")

            results["per_target"].append({
                "target": target,
                "steps": int(steps),
                "reached": bool(reached),
                "final_dist": float(dist),
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    traj = np.array(results["trajectory"])
    ax1.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2, label="Trajectory")
    ax1.plot(traj[0, 0], traj[0, 1], "go", markersize=12, label="Start")
    ax1.plot(traj[-1, 0], traj[-1, 1], "rs", markersize=10, label="End")

    for i, t in enumerate(results["targets"]):
        circle = plt.Circle(t, 0.12, fill=False, color="r", linestyle="--", linewidth=1.5)
        ax1.add_patch(circle)
        ax1.plot(t[0], t[1], "r*", markersize=12)
        ax1.annotate(f"{i + 1}", (t[0] + 0.08, t[1] + 0.08), fontsize=10, color="red")

    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-1.3, 1.3)
    ax1.set_xlabel("Y Position (left-right)", fontsize=11)
    ax1.set_ylabel("Z Position (up-down)", fontsize=11)
    ax1.set_title("2D Trajectory", fontsize=12)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    ax2 = axes[1]
    timestamps = np.array(results["timestamps"])
    ax2.plot(timestamps, traj[:, 0], "b-", linewidth=2.5, label="Y (left-right)")
    ax2.plot(timestamps, traj[:, 1], "g--", linewidth=2, marker="o", markersize=3,
             markevery=max(1, len(timestamps) // 15), label="Z (up-down)")

    for t in results["targets"]:
        ax2.axhline(y=t[0], color="b", linestyle=":", alpha=0.5, linewidth=1.5)
        ax2.axhline(y=t[1], color="g", linestyle=":", alpha=0.5, linewidth=1.5)

    ax2.set_xlabel("Time (s)", fontsize=11)
    ax2.set_ylabel("Position", fontsize=11)
    ax2.set_title("Position vs Time", fontsize=12)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(
        f"{results['description']}\n"
        f"Total Steps: {results['total_steps']}  |  max_steps/target: {results['max_steps']}",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_patterns(all_results: Dict[str, Dict], output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    pattern_order = list(TEST_PATTERNS.keys())
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
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", fontsize=14)
            ax.set_title(labels.get(name, name))
            continue

        results = all_results[name]
        traj = np.array(results["trajectory"])
        timestamps = np.array(results["timestamps"])

        ax.plot(timestamps, traj[:, 0], "b-", linewidth=2.5, label="Y (left-right)")
        ax.plot(timestamps, traj[:, 1], "g--", linewidth=2, marker="o", markersize=2,
                markevery=max(1, len(timestamps) // 10), label="Z (up-down)")

        for t in results["targets"]:
            ax.axhline(y=t[0], color="b", linestyle=":", alpha=0.4)
            ax.axhline(y=t[1], color="g", linestyle=":", alpha=0.4)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.set_title(f"{labels.get(name, name)} ({results['total_steps']} steps)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "8-Direction Comprehensive Test V2: Position vs Time\n"
        "(Larger amplitude: smaller norm_step, more steps per target, no soft limits)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n综合图已保存: {output_path}")


def plot_all_trajectories(all_results: Dict[str, Dict], output_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    pattern_order = list(TEST_PATTERNS.keys())
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
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_title(labels.get(name, name))
            continue

        results = all_results[name]
        traj = np.array(results["trajectory"])

        ax.plot(traj[:, 0], traj[:, 1], "b-", linewidth=2)
        ax.plot(traj[0, 0], traj[0, 1], "go", markersize=10, label="Start")

        for i, t in enumerate(results["targets"]):
            circle = plt.Circle(t, 0.12, fill=False, color="r", linestyle="--", linewidth=1.5)
            ax.add_patch(circle)
            ax.plot(t[0], t[1], "r*", markersize=10)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_xlabel("Y")
        ax.set_ylabel("Z")
        ax.set_title(labels.get(name, name))
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.suptitle(
        "8-Direction Comprehensive Test V2: 2D Trajectories",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"轨迹图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="8 方向综合运动测试 V2 (更大幅度)")

    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--step-rad", type=float, default=0.70,
                   help="每步弧度 (不变, 保持单步平滑)")
    p.add_argument("--norm-step", type=float, default=0.10,
                   help="归一化步长 (减小→需要更多步→更大物理运动幅度)")
    p.add_argument("--max-steps", type=int, default=50,
                   help="每个目标的最大步数")
    p.add_argument("--velocity", type=int, default=140,
                   help="测试运动速度")
    p.add_argument("--repeat", type=int, default=2,
                   help="每个动作重复次数")
    p.add_argument("--target-radius", type=float, default=0.12,
                   help="目标到达判定半径")

    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path, default=_ROOT / "serial_return.json")

    p.add_argument("--output-dir", type=str,
                   default="outputs/8dir_comprehensive_test_v2")

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("8 方向综合运动测试 V2 (更大运动幅度)")
    print("=" * 70)
    print(f"串口: {args.serial_port}")
    print(f"步长: step_rad={args.step_rad}, norm_step={args.norm_step}")
    print(f"每目标最大步数: {args.max_steps}")
    print(f"测试速度: {args.velocity}")
    print(f"软限位: 已移除 (soft_limit_margin=0.0)")
    print()
    print(f"V1 对比: norm_step 0.25→{args.norm_step}, "
          f"max_steps 30→{args.max_steps}")
    print(f"  → 步数增加 {args.max_steps / 30:.1f}x, "
          f"物理幅度增加约 {0.25 / args.norm_step:.1f}x")
    print()

    # 同步归中
    go_to_home(args.serial_port, args.home_json, duration=2.0)

    # 初始化环境 (soft_limit_margin=0.0 移除软限位)
    cfg = SerialConfigV2(
        port=args.serial_port,
        move_velocity=args.velocity,
        move_time_ms=2000,
        action_delay_ms=0,
        joint_step_rad=args.step_rad,
        soft_limit_margin=0.0,
    )

    try:
        serial_env = SerialArmEnvV2(cfg)
    except Exception as e:
        print(f"[错误] 无法初始化串口: {e}")
        return

    controller = SmoothMotionControllerV2(
        serial_env=serial_env,
        step_rad=args.step_rad,
        velocity=args.velocity,
        norm_step=args.norm_step,
    )

    all_results = {}

    try:
        print(f"\n{'=' * 60}")
        print(f"测试开始 [ velocity={args.velocity}, max_steps={args.max_steps} ]")
        print(f"{'=' * 60}")

        pattern_names = list(TEST_PATTERNS.keys())

        for i, name in enumerate(pattern_names):
            pattern = TEST_PATTERNS[name]

            results = run_single_pattern(
                controller=controller,
                pattern=pattern,
                target_radius=args.target_radius,
                max_steps=args.max_steps,
                repeat=args.repeat,
            )

            all_results[name] = results

            result_file = output_dir / f"{name}.json"
            with open(result_file, "w") as f:
                save_data = {
                    "name": results["name"],
                    "description": results["description"],
                    "targets": [list(t) for t in results["targets"]],
                    "max_steps": results["max_steps"],
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

            plot_single_pattern(results, output_dir / f"{name}.png")
            print(f"  图像已保存: {output_dir / f'{name}.png'}")

            if i < len(pattern_names) - 1:
                controller.return_to_center()

        print(f"\n{'=' * 60}")
        print("测试结束")
        print(f"{'=' * 60}")

        plot_all_patterns(all_results, output_dir / "all_pos_vs_time.png")
        plot_all_trajectories(all_results, output_dir / "all_trajectories.png")

        summary = {
            "version": "v2",
            "norm_step": args.norm_step,
            "max_steps_per_target": args.max_steps,
            "soft_limit_margin": 0.0,
            "total_patterns": len(all_results),
            "patterns": list(all_results.keys()),
            "total_steps": sum(r["total_steps"] for r in all_results.values()),
            "total_targets_reached": sum(r["reached"] for r in all_results.values()),
            "total_targets": sum(
                len(r["per_target"]) for r in all_results.values()
            ),
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 70}")
        print("测试完成!")
        print(f"  模式: {summary['total_patterns']}")
        print(f"  总步数: {summary['total_steps']}")
        print(f"  目标到达率: "
              f"{summary['total_targets_reached']}/{summary['total_targets']}")
        print(f"  结果: {output_dir}")
        print(f"{'=' * 70}")

    finally:
        serial_env.close()
        time.sleep(0.3)
        go_to_home(args.serial_port, args.home_json, duration=2.0)
        go_to_return(args.serial_port, args.return_json, duration=2.0)


if __name__ == "__main__":
    main()
