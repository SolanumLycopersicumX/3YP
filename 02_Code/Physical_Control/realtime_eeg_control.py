#!/usr/bin/env python3
"""
实时 EEG 信号控制脚本 (OpenBCI / BrainFlow)

完整流程:
    1. 通过 BrainFlow 读取实时 EEG 数据 (或合成数据)
    2. CTNet 分类: EEG epoch → 4类 MI 预测 (Left/Right/Feet/Tongue)
    3. MI 类别 → 8方向动作映射
    4. RL 策略决策 → 执行动作 (仿真或物理机械臂)
    5. 生成 Position vs Time 可视化

模式:
    --mode sim       : 合成 EEG + 仿真环境 (纯软件演示)
    --mode physical  : 真实 OpenBCI + 物理机械臂
    --mode synthetic : 合成 EEG + 物理机械臂

由于 NUS 政策原因，不使用真人的实时 EEG 信号进行实际控制。
SYNTHETIC_BOARD 模式可以完整演示整个数据流管线。

用法:
    # 纯软件仿真 (无需硬件)
    python scripts/realtime_eeg_control.py --mode sim --steps 20

    # 合成 EEG + 物理机械臂
    python scripts/realtime_eeg_control.py --mode synthetic --serial-port /dev/ttyACM1

    # 真实 OpenBCI + 物理机械臂
    python scripts/realtime_eeg_control.py --mode physical --serial-port /dev/ttyACM1 \\
        --openbci-port /dev/ttyUSB0 --board cyton

创建时间: 2026-03-10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 项目路径
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ============================================================================
# MI 类别 → 动作映射
# ============================================================================

# CTNet 4-class MI 分类结果 → 8方向动作的映射
# Left Hand (0) → left (0)
# Right Hand (1) → right (1)
# Both Feet (2)  → down (3)
# Tongue (3)     → up (2)
MI_TO_ACTION = {
    0: 0,  # Left Hand → left
    1: 1,  # Right Hand → right
    2: 3,  # Both Feet → down
    3: 2,  # Tongue → up
}

MI_CLASS_NAMES = {
    0: "Left Hand",
    1: "Right Hand",
    2: "Both Feet",
    3: "Tongue",
}

ACTION_NAMES_8 = [
    "left", "right", "up", "down",
    "up_left", "up_right", "down_left", "down_right"
]

ACTION_VECTORS_8 = {
    0: (-1.0, 0.0),    # left
    1: (1.0, 0.0),     # right
    2: (0.0, 1.0),     # up
    3: (0.0, -1.0),    # down
    4: (-0.707, 0.707),   # up_left
    5: (0.707, 0.707),    # up_right
    6: (-0.707, -0.707),  # down_left
    7: (0.707, -0.707),   # down_right
}


# ============================================================================
# CTNet 分类器封装
# ============================================================================

class CTNetClassifier:
    """
    CTNet EEG 分类器封装
    
    加载训练好的 CTNet 模型，对 EEG epoch 进行 4-class MI 分类。
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"CTNet 模型不存在: {model_path}")
        
        # CTNet 模型使用 torch.save(model) 保存了完整模型 (非 state_dict),
        # 且保存时 CTNet_model.py 是作为 __main__ 运行的，
        # 所以 pickle 存储的类路径是 __main__.EEGTransformer 等。
        # 需要将 CTNet_model 的所有类注入当前 __main__ 命名空间。
        import CTNet_model
        import __main__
        for attr_name in dir(CTNet_model):
            if not attr_name.startswith('_'):
                setattr(__main__, attr_name, getattr(CTNet_model, attr_name))
        
        self.model = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.eval()
        print(f"[CTNetClassifier] ✓ 模型已加载: {model_path}")
        print(f"[CTNetClassifier]   设备: {self.device}")
    
    def predict(self, epoch: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        对单个 EEG epoch 进行分类
        
        Parameters
        ----------
        epoch : ndarray, shape (1, 1, channels, samples)
            CTNet 格式的 EEG 数据
        
        Returns
        -------
        pred_class : int
            预测类别 (0=Left, 1=Right, 2=Feet, 3=Tongue)
        probabilities : ndarray, shape (4,)
            各类别概率
        """
        with torch.no_grad():
            x = torch.from_numpy(epoch).float().to(self.device)
            _, logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))
        
        return pred_class, probs


# ============================================================================
# 仿真环境 (无需物理硬件)
# ============================================================================

class SimulatedArmEnv:
    """
    仿真机械臂环境
    
    模拟 2D 平面上的移动，无需物理硬件。
    """
    
    def __init__(self, norm_step: float = 0.15):
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
        """执行 8 方向动作"""
        if action not in ACTION_VECTORS_8:
            return (self._y, self._z)
        
        dy, dz = ACTION_VECTORS_8[action]
        self._y = np.clip(self._y + dy * self.norm_step, -1.0, 1.0)
        self._z = np.clip(self._z + dz * self.norm_step, -1.0, 1.0)
        
        self.trajectory.append((self._y, self._z))
        self.timestamps.append(time.time() - self._start_time)
        
        return (self._y, self._z)
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self._y, self._z)
    
    def close(self):
        pass


# ============================================================================
# 实时控制主循环
# ============================================================================

@dataclass
class ControlStep:
    """单步控制记录"""
    step: int
    eeg_class: int
    eeg_class_name: str
    eeg_confidence: float
    action: int
    action_name: str
    position: Tuple[float, float]
    timestamp: float


def run_realtime_control(
    eeg_stream,                      # OpenBCIStream 实例
    classifier: CTNetClassifier,     # CTNet 分类器
    arm_env,                         # 机械臂环境 (仿真或物理)
    max_steps: int = 20,
    epoch_duration: float = 4.0,     # 每次 EEG 采集的时长
    use_rl: bool = False,            # 是否使用 RL 策略 (否则直接用 MI→action 映射)
    rl_model = None,                 # RL 模型
    rl_device = None,                # RL 设备
    verbose: bool = True,
) -> List[ControlStep]:
    """
    实时 EEG 控制主循环
    
    每一步:
    1. 从 EEG 流中读取一段 epoch
    2. CTNet 分类 → MI 类别
    3. MI 类别 → 动作
    4. 执行动作
    
    Parameters
    ----------
    eeg_stream : OpenBCIStream
        EEG 数据流
    classifier : CTNetClassifier
        CTNet 分类器
    arm_env : SimulatedArmEnv 或物理环境
        机械臂环境
    max_steps : int
        最大步数
    epoch_duration : float
        每次 EEG 采集时长 (秒)
    use_rl : bool
        是否使用 RL 策略决策
    rl_model : nn.Module or None
        RL 模型
    rl_device : torch.device or None
        RL 运行设备
    verbose : bool
        是否打印详细信息
    
    Returns
    -------
    history : List[ControlStep]
        控制历史记录
    """
    history: List[ControlStep] = []
    start_time = time.time()
    
    if verbose:
        print("\n" + "=" * 70)
        print("实时 EEG 控制 — 开始")
        print("=" * 70)
        print(f"  最大步数: {max_steps}")
        print(f"  EEG epoch 时长: {epoch_duration}s")
        print(f"  控制模式: {'RL 策略' if use_rl else 'MI 直接映射'}")
        print()
    
    for step_idx in range(max_steps):
        # 1. 读取 EEG epoch
        if verbose:
            print(f"[Step {step_idx + 1}/{max_steps}] 采集 EEG 数据 ({epoch_duration}s)...", 
                  end="", flush=True)
        
        epoch = eeg_stream.get_eeg_epoch_for_ctnet(duration_sec=epoch_duration)
        
        # 2. CTNet 分类
        mi_class, probs = classifier.predict(epoch)
        confidence = probs[mi_class]
        
        # 3. 映射到动作
        if use_rl and rl_model is not None:
            # RL 策略: 使用当前状态决策
            y, z = arm_env.position
            # 这里简化为固定目标 (0.5, 0.5)
            target_y, target_z = 0.5, 0.5
            dist = np.sqrt((y - target_y)**2 + (z - target_z)**2)
            state = np.array([y, z, target_y, target_z, dist], dtype=np.float32)
            state_tensor = torch.tensor(state).unsqueeze(0).to(rl_device)
            
            with torch.no_grad():
                q_values = rl_model(state_tensor)
                action = q_values.argmax(dim=1).item()
        else:
            # 直接映射: MI 类别 → 8方向动作
            action = MI_TO_ACTION.get(mi_class, 0)
        
        action_name = ACTION_NAMES_8[action] if action < len(ACTION_NAMES_8) else f"unknown({action})"
        
        # 4. 执行动作
        pos = arm_env.execute_action(action)
        
        # 记录
        step_record = ControlStep(
            step=step_idx + 1,
            eeg_class=mi_class,
            eeg_class_name=MI_CLASS_NAMES[mi_class],
            eeg_confidence=float(confidence),
            action=action,
            action_name=action_name,
            position=pos,
            timestamp=time.time() - start_time,
        )
        history.append(step_record)
        
        if verbose:
            print(f" EEG={MI_CLASS_NAMES[mi_class]} ({confidence:.1%}) "
                  f"→ {action_name} → pos=({pos[0]:.2f}, {pos[1]:.2f})")
    
    # 总结
    if verbose:
        print("\n" + "-" * 40)
        print(f"控制完成: {len(history)} 步")
        unique_classes = set(s.eeg_class for s in history)
        for cls in sorted(unique_classes):
            count = sum(1 for s in history if s.eeg_class == cls)
            print(f"  {MI_CLASS_NAMES[cls]}: {count} 次 ({count/len(history):.0%})")
        final_pos = history[-1].position
        print(f"  最终位置: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        print("-" * 40)
    
    return history


# ============================================================================
# 可视化
# ============================================================================

def plot_control_results(
    history: List[ControlStep],
    output_path: Path,
    title: str = "Real-time EEG Control",
):
    """绘制控制结果"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    steps = [s.step for s in history]
    timestamps = [s.timestamp for s in history]
    y_pos = [s.position[0] for s in history]
    z_pos = [s.position[1] for s in history]
    classes = [s.eeg_class for s in history]
    confidences = [s.eeg_confidence for s in history]
    
    # 1. Position vs Time
    ax1 = axes[0, 0]
    ax1.plot(timestamps, y_pos, 'b-', linewidth=2, marker='o', markersize=3, label='Y (left-right)')
    ax1.plot(timestamps, z_pos, 'g--', linewidth=2, marker='s', markersize=3, label='Z (up-down)')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Position', fontsize=11)
    ax1.set_title('Position vs Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-1.1, 1.1)
    
    # 2. 2D 轨迹
    ax2 = axes[0, 1]
    ax2.plot(y_pos, z_pos, 'b-', linewidth=2, alpha=0.7)
    ax2.scatter([y_pos[0]], [z_pos[0]], c='green', s=100, zorder=5, label='Start')
    ax2.scatter([y_pos[-1]], [z_pos[-1]], c='red', s=100, zorder=5, marker='x', label='End')
    ax2.set_xlabel('Y (left-right)', fontsize=11)
    ax2.set_ylabel('Z (up-down)', fontsize=11)
    ax2.set_title('2D Trajectory', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-1.1, 1.1)
    ax2.set_ylim(-1.1, 1.1)
    ax2.set_aspect('equal')
    
    # 3. EEG 分类结果
    ax3 = axes[1, 0]
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800']
    class_colors = [colors[c] for c in classes]
    ax3.bar(steps, confidences, color=class_colors, alpha=0.8)
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel('Classification Confidence', fontsize=11)
    ax3.set_title('EEG Classification per Step', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 图例
    for i, name in MI_CLASS_NAMES.items():
        ax3.bar([], [], color=colors[i], label=name)
    ax3.legend(loc='upper right', fontsize=8)
    
    # 4. 类别分布
    ax4 = axes[1, 1]
    class_counts = [sum(1 for c in classes if c == i) for i in range(4)]
    bars = ax4.bar(
        [MI_CLASS_NAMES[i] for i in range(4)],
        class_counts,
        color=colors,
        alpha=0.8,
    )
    for bar, count in zip(bars, class_counts):
        if count > 0:
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax4.set_xlabel('MI Class', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Classification Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  结果图已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="实时 EEG 信号控制 (OpenBCI / BrainFlow)"
    )
    
    # 运行模式
    p.add_argument("--mode", choices=["sim", "synthetic", "physical"],
                   default="sim",
                   help="运行模式: sim (纯仿真), synthetic (合成EEG+物理臂), "
                        "physical (真实EEG+物理臂)")
    
    # EEG 参数
    p.add_argument("--board", type=str, default="synthetic",
                   choices=["synthetic", "cyton", "cyton_daisy", "ganglion"],
                   help="OpenBCI 板卡类型")
    p.add_argument("--openbci-port", type=str, default=None,
                   help="OpenBCI 串口 (真实硬件时必需)")
    p.add_argument("--epoch-duration", type=float, default=4.0,
                   help="每次 EEG 采集时长 (秒)")
    
    # CTNet 参数
    p.add_argument("--ctnet-model", type=str,
                   default=str(_ROOT / "A_heads_2_depth_6" / "model_1.pth"),
                   help="CTNet 模型路径")
    
    # 控制参数
    p.add_argument("--steps", type=int, default=20,
                   help="最大控制步数")
    p.add_argument("--norm-step", type=float, default=0.15,
                   help="归一化移动步长 (增大 → 更大的运动幅度)")
    p.add_argument("--step-rad", type=float, default=0.50,
                   help="物理臂每步弧度 (增大 → 更大的物理运动)")
    p.add_argument("--velocity", type=int, default=100,
                   help="物理臂速度")
    
    # 物理臂参数
    p.add_argument("--serial-port", type=str, default="/dev/ttyACM1",
                   help="物理机械臂串口")
    
    # RL 参数
    p.add_argument("--use-rl", action="store_true",
                   help="使用 RL 策略而非直接映射")
    p.add_argument("--rl-model", type=str,
                   default=str(_ROOT / "outputs" / "rl_8direction_smooth" /
                              "rl_8direction_smooth_transformer.pth"),
                   help="RL 模型路径")
    
    # 输出参数
    p.add_argument("--output-dir", type=Path,
                   default=_ROOT / "outputs" / "realtime_eeg_control")
    p.add_argument("--no-plot", action="store_true",
                   help="不生成结果图")
    
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("实时 EEG 信号控制 (OpenBCI / BrainFlow)")
    print("=" * 70)
    print(f"  模式: {args.mode}")
    print(f"  板卡: {args.board}")
    print(f"  CTNet 模型: {args.ctnet_model}")
    print(f"  步数: {args.steps}")
    print(f"  移动步长: norm_step={args.norm_step}, step_rad={args.step_rad}")
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ----------------------------------------------------------------
    # 1. 初始化 EEG 数据流
    # ----------------------------------------------------------------
    print("[1/4] 初始化 EEG 数据流...")
    from scripts.openbci_stream import OpenBCIStream
    
    if args.mode == "physical":
        # 真实硬件
        eeg_stream = OpenBCIStream(
            board_type=args.board,
            serial_port=args.openbci_port,
        )
    else:
        # 合成板卡 (sim 和 synthetic 模式都用合成EEG)
        eeg_stream = OpenBCIStream(board_type="synthetic")
    
    # ----------------------------------------------------------------
    # 2. 初始化 CTNet 分类器
    # ----------------------------------------------------------------
    print("\n[2/4] 初始化 CTNet 分类器...")
    classifier = CTNetClassifier(
        model_path=args.ctnet_model,
        device=device,
    )
    
    # ----------------------------------------------------------------
    # 3. 初始化机械臂环境
    # ----------------------------------------------------------------
    print("\n[3/4] 初始化机械臂环境...")
    arm_env = None
    
    if args.mode == "sim":
        # 仿真环境
        arm_env = SimulatedArmEnv(norm_step=args.norm_step)
        print(f"  使用仿真环境 (norm_step={args.norm_step})")
    else:
        # 物理机械臂
        from serial_arm_env_v2 import SerialArmEnvV2, SerialConfigV2
        
        cfg = SerialConfigV2(
            port=args.serial_port,
            move_velocity=args.velocity,
            move_time_ms=2000,
            action_delay_ms=0,
            joint_step_rad=args.step_rad,
        )
        
        try:
            serial_env = SerialArmEnvV2(cfg)
        except Exception as e:
            print(f"[错误] 无法初始化物理机械臂: {e}")
            print("[回退] 使用仿真环境")
            arm_env = SimulatedArmEnv(norm_step=args.norm_step)
        
        if arm_env is None:
            # 物理臂控制器 (类似 eeg_physical_control.py 中的控制器, 但独立)
            from scripts.eeg_physical_control import EEGPhysicalController
            arm_env = EEGPhysicalController(
                serial_env=serial_env,
                step_rad=args.step_rad,
                velocity=args.velocity,
                norm_step=args.norm_step,
                noise_level=0.0,  # 实时模式不加噪声
            )
    
    # RL 模型 (可选)
    rl_model = None
    if args.use_rl:
        rl_model_path = Path(args.rl_model)
        if rl_model_path.exists():
            from scripts.train_rl_8direction_smooth import SequenceTransformerDQN
            rl_model = SequenceTransformerDQN(state_dim=5, action_dim=8, seq_len=10)
            state = torch.load(rl_model_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            rl_model.load_state_dict(state)
            rl_model = rl_model.to(device)
            rl_model.eval()
            print(f"[RL] ✓ 已加载 RL 模型: {rl_model_path}")
        else:
            print(f"[RL] 模型不存在: {rl_model_path}, 使用直接映射")
    
    # ----------------------------------------------------------------
    # 4. 运行控制循环
    # ----------------------------------------------------------------
    print("\n[4/4] 开始实时控制...")
    
    eeg_stream.start()
    
    try:
        history = run_realtime_control(
            eeg_stream=eeg_stream,
            classifier=classifier,
            arm_env=arm_env,
            max_steps=args.steps,
            epoch_duration=args.epoch_duration,
            use_rl=args.use_rl,
            rl_model=rl_model,
            rl_device=device,
            verbose=True,
        )
        
        # 保存结果
        results_json = {
            "mode": args.mode,
            "board": args.board,
            "steps": len(history),
            "norm_step": args.norm_step,
            "step_rad": args.step_rad,
            "epoch_duration": args.epoch_duration,
            "history": [
                {
                    "step": s.step,
                    "eeg_class": s.eeg_class,
                    "eeg_class_name": s.eeg_class_name,
                    "eeg_confidence": s.eeg_confidence,
                    "action": s.action,
                    "action_name": s.action_name,
                    "position": list(s.position),
                    "timestamp": s.timestamp,
                }
                for s in history
            ],
        }
        
        json_path = output_dir / f"realtime_control_{args.mode}.json"
        with open(json_path, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\n  结果 JSON 已保存: {json_path}")
        
        # 可视化
        if not args.no_plot:
            plot_control_results(
                history=history,
                output_path=output_dir / f"realtime_control_{args.mode}.png",
                title=f"Real-time EEG Control ({args.mode.capitalize()} Mode)\n"
                      f"Steps: {len(history)}, norm_step={args.norm_step}",
            )
    
    finally:
        eeg_stream.stop()
        if hasattr(arm_env, 'close'):
            arm_env.close()
    
    print("\n" + "=" * 70)
    print("实时 EEG 控制完成!")
    print(f"结果保存: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
