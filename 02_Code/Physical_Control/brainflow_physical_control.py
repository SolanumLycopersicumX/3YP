#!/usr/bin/env python3
"""
BrainFlow 实时 EEG + 物理机械臂控制

完整 BCI 闭环流程:
    1. BrainFlow 实时读取 EEG 数据 (OpenBCI 合成板 / Cyton / Ganglion)
    2. CTNet 对 EEG epoch 进行 4-class MI 分类
    3. MI 类别映射到 8 方向动作
    4. 物理机械臂 (SO-101) 执行动作
    5. 生成 Position vs Time 可视化 & 对比分析

运行模式:
    --mode demo      : 合成 EEG + 物理机械臂 (默认, 使用最优动作引导,
                       同时运行 BrainFlow+CTNet 演示数据管线)
    --mode free      : 合成 EEG + 物理机械臂 (完全由 CTNet 分类驱动)
    --mode sim       : 合成 EEG + 仿真环境   (无需硬件)

板卡类型:
    --board synthetic : BrainFlow SYNTHETIC_BOARD (默认, 无需硬件)
    --board cyton     : OpenBCI Cyton  (需 --openbci-port)
    --board ganglion  : OpenBCI Ganglion (需 --openbci-port)

由于 NUS 政策原因, 不使用真人的实时 EEG 信号进行实际控制。
SYNTHETIC_BOARD 模式可以完整演示整个数据流管线。

用法:
    # 合成 EEG + 物理机械臂 (演示模式, 使用最优动作引导)
    python scripts/brainflow_physical_control.py \\
        --serial-port /dev/ttyACM1

    # 合成 EEG + 仿真环境 (无需硬件)
    python scripts/brainflow_physical_control.py --mode sim

    # 真实 OpenBCI Cyton + 物理机械臂
    python scripts/brainflow_physical_control.py \\
        --serial-port /dev/ttyACM1 \\
        --board cyton --openbci-port /dev/ttyUSB0

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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# MI 类别 → 动作映射
# ============================================================================

MI_CLASS_NAMES = {0: "Left Hand", 1: "Right Hand", 2: "Both Feet", 3: "Tongue"}

MI_TO_ACTION = {0: 0, 1: 1, 2: 3, 3: 2}

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
        "a_horizontal", "Horizontal (Left→Right→Center)",
        [(-1.00, 0.0), (1.00, 0.0), (0.0, 0.0)],
    ),
    "b_vertical": TestPattern(
        "b_vertical", "Vertical (Up→Down→Center)",
        [(0.0, 1.00), (0.0, -1.00), (0.0, 0.0)],
    ),
    "c_diagonal_ul_dr": TestPattern(
        "c_diagonal_ul_dr", "Diagonal (UL→DR)",
        [(-0.85, 0.85), (0.85, -0.85), (0.0, 0.0)],
    ),
    "d_diagonal_ur_dl": TestPattern(
        "d_diagonal_ur_dl", "Diagonal (UR→DL)",
        [(0.85, 0.85), (-0.85, -0.85), (0.0, 0.0)],
    ),
    "e_square_cw": TestPattern(
        "e_square_cw", "Square (CW)",
        [(0.85, 0.0), (0.85, -0.85), (-0.85, -0.85),
         (-0.85, 0.85), (0.85, 0.85), (0.0, 0.0)],
    ),
    "f_square_ccw": TestPattern(
        "f_square_ccw", "Square (CCW)",
        [(0.0, 0.85), (0.85, 0.85), (0.85, -0.85),
         (-0.85, -0.85), (-0.85, 0.85), (0.0, 0.0)],
    ),
}


# ============================================================================
# CTNet 分类器
# ============================================================================

class CTNetClassifier:
    """加载训练好的 CTNet 模型, 对 EEG epoch 进行 4-class MI 分类。"""

    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        if not Path(model_path).exists():
            raise FileNotFoundError(f"CTNet model not found: {model_path}")

        import CTNet_model
        import __main__
        for attr_name in dir(CTNet_model):
            if not attr_name.startswith("_"):
                setattr(__main__, attr_name, getattr(CTNet_model, attr_name))

        self.model = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.eval()
        print(f"[CTNet] Loaded: {model_path}  Device: {self.device}")

    def predict(self, epoch: np.ndarray) -> Tuple[int, np.ndarray]:
        with torch.no_grad():
            x = torch.from_numpy(epoch).float().to(self.device)
            _, logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            return int(np.argmax(probs)), probs


class DummyClassifier:
    """当 CTNet 模型不可用时, 用随机分类代替。"""

    def predict(self, epoch: np.ndarray) -> Tuple[int, np.ndarray]:
        probs = np.random.dirichlet([1, 1, 1, 1])
        return int(np.argmax(probs)), probs


# ============================================================================
# 仿真环境
# ============================================================================

class SimulatedArmEnv:
    def __init__(self, norm_step: float = 0.10):
        self.norm_step = norm_step
        self._y = 0.0
        self._z = 0.0
        self.trajectory: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.timestamps: List[float] = [0.0]
        self._start_time = time.time()

    def reset(self):
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self._start_time = time.time()

    def execute_action(self, action: int) -> Tuple[float, float]:
        if action not in ACTION_VECTORS:
            return (self._y, self._z)
        dy, dz = ACTION_VECTORS[action]
        self._y = np.clip(self._y + dy * self.norm_step, -1.5, 1.5)
        self._z = np.clip(self._z + dz * self.norm_step, -1.5, 1.5)
        self.trajectory.append((self._y, self._z))
        self.timestamps.append(time.time() - self._start_time)
        return (self._y, self._z)

    @property
    def position(self) -> Tuple[float, float]:
        return (self._y, self._z)

    def close(self):
        pass


# ============================================================================
# 物理机械臂控制器
# ============================================================================

class PhysicalArmController:
    """
    物理机械臂控制器 (V2 参数)

    - 减小 norm_step (0.10) 使每个目标需要更多步 → 更大物理幅度
    - 移除软限位 (由 SerialConfigV2 的 soft_limit_margin=0 控制)
    """

    def __init__(
        self,
        serial_env,
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
        self._set_velocity()

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

    def _set_velocity(self):
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
        except Exception as e:
            print(f"[警告] 设置速度失败: {e}")

    def reset(self):
        self._y = 0.0
        self._z = 0.0
        self.trajectory = [(0.0, 0.0)]
        self.timestamps = [0.0]
        self._start_time = time.time()
        self._set_velocity()

    def _clamp_to_hw_limits(self, jid: int, ticks: int) -> int:
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

        from drivers.so101_serial import So101Bus

        d_lr_ticks = So101Bus.rad_to_ticks(-eff_dy * self.step_rad)
        d_ud_ticks = So101Bus.rad_to_ticks(eff_dz * self.step_rad)

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

    @property
    def position(self) -> Tuple[float, float]:
        return (self._y, self._z)

    def return_to_center(self):
        id_lr = self.serial_env._id_lr
        id_ud = self.serial_env._id_ud
        center_lr = self.serial_env._joint_centers.get(id_lr)
        center_ud = self.serial_env._joint_centers.get(id_ud)
        if center_lr is None or center_ud is None:
            return
        try:
            bus = self.serial_env._bus
            bus.write_position(id_lr, center_lr)
            bus.write_position(id_ud, center_ud)
            time.sleep(1.5)
        except Exception:
            pass
        self._y = 0.0
        self._z = 0.0
        self.trajectory.append((0.0, 0.0))
        self.timestamps.append(time.time() - self._start_time)


# ============================================================================
# 获取最优动作 (引导模式)
# ============================================================================

def get_optimal_action(
    y: float, z: float, ty: float, tz: float,
) -> int:
    dy = ty - y
    dz = tz - z
    direction = np.array([dy, dz])
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return 0
    direction = direction / norm

    best_action = 0
    best_dot = -float("inf")
    for action, vec in ACTION_VECTORS.items():
        vec_n = np.array(vec) / np.linalg.norm(vec)
        dot = np.dot(direction, vec_n)
        if dot > best_dot:
            best_dot = dot
            best_action = action
    return best_action


# ============================================================================
# 同步归位
# ============================================================================

def go_to_home(serial_port: str, home_json: Path, duration: float = 2.0):
    if not home_json.exists():
        return
    print("\n[同步回中位]")
    os.system(
        f"python {_HERE}/serial_go_home_sync.py "
        f"--port {serial_port} --home-json {home_json} "
        f"--duration {duration} --wait 1.5"
    )
    time.sleep(0.5)


def go_to_return(serial_port: str, return_json: Path, duration: float = 2.0):
    if not return_json.exists():
        return
    print("\n[同步归位]")
    os.system(
        f"python {_HERE}/serial_go_return_sync.py "
        f"--port {serial_port} --return-json {return_json} "
        f"--duration {duration} --wait 1.5"
    )


# ============================================================================
# 主控制循环
# ============================================================================

@dataclass
class StepRecord:
    step: int
    eeg_class: int
    eeg_class_name: str
    eeg_confidence: float
    action_used: int
    action_name: str
    position: Tuple[float, float]
    timestamp: float
    guided: bool


def run_brainflow_pattern(
    eeg_stream,
    classifier,
    arm_env,
    pattern: TestPattern,
    *,
    guided: bool = True,
    max_steps: int = 50,
    target_radius: float = 0.12,
    epoch_sec: float = 1.0,
) -> Tuple[List[StepRecord], Dict]:
    """
    使用 BrainFlow EEG 数据运行单个测试模式。

    在 guided=True (demo) 模式下:
    - 每步都读取 BrainFlow 数据并用 CTNet 分类 (演示数据管线)
    - 但实际动作使用最优方向引导 (确保有意义的运动轨迹)

    在 guided=False (free) 模式下:
    - 完全由 CTNet 分类结果驱动动作
    """
    history: List[StepRecord] = []

    print(f"\n{'=' * 60}")
    print(f"Pattern: {pattern.description}  "
          f"{'[GUIDED]' if guided else '[FREE]'}")
    print(f"Targets: {pattern.targets}")
    print(f"{'=' * 60}")

    arm_env.reset()
    total_steps_used = 0
    reached = 0

    for t_idx, (ty, tz) in enumerate(pattern.targets):
        steps = 0
        print(f"  Target {t_idx + 1}/{len(pattern.targets)} "
              f"({ty:.2f}, {tz:.2f})", end="", flush=True)

        while steps < max_steps:
            y, z = arm_env.position
            dist = np.sqrt((y - ty) ** 2 + (z - tz) ** 2)
            if dist < target_radius:
                reached += 1
                break

            # --- BrainFlow: read & classify ---
            epoch = eeg_stream.get_eeg_epoch_for_ctnet(
                duration_sec=epoch_sec,
            )
            mi_class, probs = classifier.predict(epoch)

            # --- choose action ---
            if guided:
                action = get_optimal_action(y, z, ty, tz)
            else:
                action = MI_TO_ACTION.get(mi_class, 0)

            action_name = ACTION_NAMES.get(action, f"a{action}")

            # --- execute ---
            pos = arm_env.execute_action(action)
            steps += 1
            total_steps_used += 1

            history.append(StepRecord(
                step=total_steps_used,
                eeg_class=mi_class,
                eeg_class_name=MI_CLASS_NAMES[mi_class],
                eeg_confidence=float(probs[mi_class]),
                action_used=action,
                action_name=action_name,
                position=pos,
                timestamp=arm_env.timestamps[-1],
                guided=guided,
            ))

        final_dist = np.sqrt(
            (arm_env.position[0] - ty) ** 2
            + (arm_env.position[1] - tz) ** 2
        )
        status = "✓" if final_dist < target_radius else "✗"
        print(f" {status} steps={steps} dist={final_dist:.3f}")

    info = {
        "name": pattern.name,
        "description": pattern.description,
        "targets": pattern.targets,
        "total_steps": total_steps_used,
        "reached": reached,
        "total_targets": len(pattern.targets),
        "trajectory": list(arm_env.trajectory),
        "timestamps": list(arm_env.timestamps),
    }
    return history, info


# ============================================================================
# 可视化
# ============================================================================

def plot_pattern_with_eeg(
    history: List[StepRecord],
    info: Dict,
    output_path: Path,
):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    traj = np.array(info["trajectory"])
    ts = np.array(info["timestamps"])

    # 1) Position vs Time
    ax = axes[0, 0]
    ax.plot(ts, traj[:, 0], "b-", lw=2.5, label="Y (left-right)")
    ax.plot(ts, traj[:, 1], "g--", lw=2, marker="o", markersize=3,
            markevery=max(1, len(ts) // 15), label="Z (up-down)")
    for t in info["targets"]:
        ax.axhline(t[0], color="b", ls=":", alpha=0.4)
        ax.axhline(t[1], color="g", ls=":", alpha=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position")
    ax.set_title("Position vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) 2D Trajectory
    ax = axes[0, 1]
    ax.plot(traj[:, 0], traj[:, 1], "b-", lw=2)
    ax.plot(traj[0, 0], traj[0, 1], "go", ms=12, label="Start")
    ax.plot(traj[-1, 0], traj[-1, 1], "rs", ms=10, label="End")
    for i, t in enumerate(info["targets"]):
        circle = plt.Circle(t, 0.12, fill=False, color="r", ls="--", lw=1.5)
        ax.add_patch(circle)
        ax.plot(t[0], t[1], "r*", ms=12)
        ax.annotate(f"{i + 1}", (t[0] + 0.08, t[1] + 0.08), fontsize=10, color="red")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_title("2D Trajectory")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # 3) EEG Classification per step
    ax = axes[1, 0]
    if history:
        steps = [h.step for h in history]
        confs = [h.eeg_confidence for h in history]
        classes = [h.eeg_class for h in history]
        colors_map = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
        bar_colors = [colors_map[c] for c in classes]
        ax.bar(steps, confs, color=bar_colors, alpha=0.8, width=0.8)
        for i, name in MI_CLASS_NAMES.items():
            ax.bar([], [], color=colors_map[i], label=name)
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Confidence")
    ax.set_title("EEG Classification (BrainFlow)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis="y")

    # 4) MI class distribution
    ax = axes[1, 1]
    if history:
        counts = [sum(1 for h in history if h.eeg_class == i) for i in range(4)]
        bars = ax.bar(
            [MI_CLASS_NAMES[i] for i in range(4)],
            counts,
            color=colors_map,
            alpha=0.8,
        )
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        str(cnt), ha="center", va="bottom", fontweight="bold")
    ax.set_xlabel("MI Class")
    ax.set_ylabel("Count")
    ax.set_title("Classification Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    mode_label = "GUIDED" if (history and history[0].guided) else "FREE"
    plt.suptitle(
        f"BrainFlow EEG → CTNet → Robot Arm  [{mode_label}]\n"
        f"{info['description']}  |  "
        f"Steps: {info['total_steps']}  |  "
        f"Reached: {info['reached']}/{info['total_targets']}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_all_patterns_summary(
    all_infos: Dict[str, Dict],
    output_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    labels = {
        "a_horizontal": "a. Horizontal",
        "b_vertical": "b. Vertical",
        "c_diagonal_ul_dr": "c. UL-DR Diagonal",
        "d_diagonal_ur_dl": "d. UR-DL Diagonal",
        "e_square_cw": "e. CW Square",
        "f_square_ccw": "f. CCW Square",
    }

    for idx, name in enumerate(TEST_PATTERNS):
        ax = axes[idx]
        if name not in all_infos:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_title(labels.get(name, name))
            continue

        info = all_infos[name]
        traj = np.array(info["trajectory"])
        ts = np.array(info["timestamps"])

        ax.plot(ts, traj[:, 0], "b-", lw=2.5, label="Y")
        ax.plot(ts, traj[:, 1], "g--", lw=2, marker="o", ms=2,
                markevery=max(1, len(ts) // 10), label="Z")
        for t in info["targets"]:
            ax.axhline(t[0], color="b", ls=":", alpha=0.4)
            ax.axhline(t[1], color="g", ls=":", alpha=0.4)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position")
        ax.set_title(f"{labels.get(name, name)} ({info['total_steps']} steps)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        "BrainFlow Real-time EEG Control: All Patterns",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n综合图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BrainFlow 实时 EEG + 物理机械臂控制"
    )

    p.add_argument("--mode", choices=["demo", "free", "sim"], default="demo",
                   help="运行模式: demo (引导+EEG演示), free (CTNet驱动), "
                        "sim (仿真)")
    p.add_argument("--board", type=str, default="synthetic",
                   choices=["synthetic", "cyton", "cyton_daisy", "ganglion"],
                   help="OpenBCI 板卡类型")
    p.add_argument("--openbci-port", type=str, default=None,
                   help="OpenBCI 串口 (真实硬件时必需)")
    p.add_argument("--epoch-sec", type=float, default=1.0,
                   help="每步 EEG 采集时长 (秒, 较短→更快)")

    p.add_argument("--ctnet-model", type=str,
                   default=str(_ROOT / "A_heads_2_depth_6" / "model_1.pth"),
                   help="CTNet 模型路径")

    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1")
    p.add_argument("--step-rad", type=float, default=0.70)
    p.add_argument("--norm-step", type=float, default=0.10,
                   help="归一化步长 (减小→更多步→更大物理幅度)")
    p.add_argument("--max-steps", type=int, default=50)
    p.add_argument("--velocity", type=int, default=140)
    p.add_argument("--target-radius", type=float, default=0.12)
    p.add_argument("--repeat", type=int, default=1,
                   help="每个动作重复次数")

    p.add_argument("--home-json", type=Path, default=_ROOT / "serial_home.json")
    p.add_argument("--return-json", type=Path, default=_ROOT / "serial_return.json")
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "brainflow_physical_control")

    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BrainFlow Real-time EEG + Physical Robot Arm Control")
    print("=" * 70)
    print(f"  Mode:          {args.mode}")
    print(f"  Board:         {args.board}")
    print(f"  Epoch:         {args.epoch_sec}s")
    print(f"  CTNet model:   {args.ctnet_model}")
    print(f"  Step:          step_rad={args.step_rad}, norm_step={args.norm_step}")
    print(f"  Max steps:     {args.max_steps}")
    print(f"  Soft limits:   DISABLED (soft_limit_margin=0.0)")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------------------------------------
    # 1. BrainFlow EEG 数据流
    # ----------------------------------------------------------------
    print("[1/4] 初始化 BrainFlow EEG 数据流...")
    from scripts.openbci_stream import OpenBCIStream

    eeg_stream = OpenBCIStream(
        board_type=args.board,
        serial_port=args.openbci_port,
    )

    # ----------------------------------------------------------------
    # 2. CTNet 分类器
    # ----------------------------------------------------------------
    print("\n[2/4] 初始化 CTNet 分类器...")
    try:
        classifier = CTNetClassifier(args.ctnet_model, device)
    except FileNotFoundError:
        print(f"[警告] CTNet 模型未找到, 使用随机分类代替")
        classifier = DummyClassifier()

    # ----------------------------------------------------------------
    # 3. 机械臂环境
    # ----------------------------------------------------------------
    print("\n[3/4] 初始化机械臂环境...")
    arm_env = None
    serial_env = None

    if args.mode == "sim":
        arm_env = SimulatedArmEnv(norm_step=args.norm_step)
        print("  使用仿真环境")
    else:
        from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2

        go_to_home(args.serial_port, args.home_json, duration=2.0)

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
            arm_env = PhysicalArmController(
                serial_env=serial_env,
                step_rad=args.step_rad,
                velocity=args.velocity,
                norm_step=args.norm_step,
            )
            print(f"  物理机械臂已初始化 (port={args.serial_port})")
        except Exception as e:
            print(f"[错误] 物理机械臂初始化失败: {e}")
            print("[回退] 使用仿真环境")
            arm_env = SimulatedArmEnv(norm_step=args.norm_step)

    # ----------------------------------------------------------------
    # 4. 运行控制循环
    # ----------------------------------------------------------------
    print("\n[4/4] 开始 BrainFlow 实时控制...")

    guided = args.mode in ("demo", "sim")

    eeg_stream.start()

    all_histories: Dict[str, List[StepRecord]] = {}
    all_infos: Dict[str, Dict] = {}

    try:
        pattern_names = list(TEST_PATTERNS.keys())

        for p_idx, name in enumerate(pattern_names):
            pattern = TEST_PATTERNS[name]

            for rep in range(args.repeat):
                rep_label = f" (rep {rep + 1}/{args.repeat})" if args.repeat > 1 else ""
                print(f"\n--- Pattern {p_idx + 1}/{len(pattern_names)}"
                      f"{rep_label} ---")

                history, info = run_brainflow_pattern(
                    eeg_stream=eeg_stream,
                    classifier=classifier,
                    arm_env=arm_env,
                    pattern=pattern,
                    guided=guided,
                    max_steps=args.max_steps,
                    target_radius=args.target_radius,
                    epoch_sec=args.epoch_sec,
                )

                all_histories[name] = history
                all_infos[name] = info

                # save per-pattern
                plot_pattern_with_eeg(history, info, output_dir / f"{name}.png")
                print(f"  图像: {output_dir / f'{name}.png'}")

                with open(output_dir / f"{name}.json", "w") as f:
                    save_data = {
                        "name": info["name"],
                        "description": info["description"],
                        "targets": [list(t) for t in info["targets"]],
                        "total_steps": info["total_steps"],
                        "reached": info["reached"],
                        "total_targets": info["total_targets"],
                        "trajectory": [list(t) for t in info["trajectory"]],
                        "timestamps": info["timestamps"],
                        "mode": args.mode,
                        "board": args.board,
                        "eeg_history": [{
                            "step": h.step,
                            "eeg_class": h.eeg_class,
                            "eeg_class_name": h.eeg_class_name,
                            "eeg_confidence": float(h.eeg_confidence),
                            "action_used": h.action_used,
                            "action_name": h.action_name,
                            "position": list(h.position),
                        } for h in history],
                    }
                    json.dump(save_data, f, indent=2)

                # 模式间归中
                if p_idx < len(pattern_names) - 1 or rep < args.repeat - 1:
                    if hasattr(arm_env, "return_to_center"):
                        arm_env.return_to_center()

        # summary plot
        plot_all_patterns_summary(all_infos, output_dir / "all_pos_vs_time.png")

        # summary json
        summary = {
            "mode": args.mode,
            "board": args.board,
            "epoch_sec": args.epoch_sec,
            "norm_step": args.norm_step,
            "max_steps": args.max_steps,
            "soft_limit_margin": 0.0,
            "total_patterns": len(all_infos),
            "total_steps": sum(i["total_steps"] for i in all_infos.values()),
            "total_reached": sum(i["reached"] for i in all_infos.values()),
            "total_targets": sum(i["total_targets"] for i in all_infos.values()),
        }
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'=' * 70}")
        print("BrainFlow 实时控制完成!")
        print(f"  模式: {args.mode} | 板卡: {args.board}")
        print(f"  测试模式: {summary['total_patterns']}")
        print(f"  总步数: {summary['total_steps']}")
        print(f"  目标到达: {summary['total_reached']}/{summary['total_targets']}")
        print(f"  结果: {output_dir}")
        print(f"{'=' * 70}")

    finally:
        eeg_stream.stop()

        if serial_env is not None:
            serial_env.close()
            time.sleep(0.3)
            go_to_home(args.serial_port, args.home_json, duration=2.0)
            go_to_return(args.serial_port, args.return_json, duration=2.0)


if __name__ == "__main__":
    main()
