#!/usr/bin/env python3
"""
RL 模型控制物理机械臂

使用训练好的 RL 模型（TransformerDQN）控制 SO-101 物理机械臂。

流程：
1. 加载 EEG 数据和分类器
2. 加载训练好的 RL 模型
3. EEG 分类器预测意图
4. RL 模型根据状态输出最优动作
5. 动作发送到物理机械臂

用法：
    python scripts/rl_physical_control.py \
        --serial-port /dev/ttyACM1 \
        --subject 1 \
        --dataset A \
        --num-trials 10

创建时间: 2026-02-17
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
import torch.nn as nn
import torch.nn.functional as F

# 项目路径设置
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from drivers.so101_serial import So101Bus, So101Map
from serial_arm_env import SerialArmEnv, SerialConfig
from scripts.dqn_transformer import TransformerDQN, LightTransformerDQN
from scripts.dqn_model import DQNNetwork


# ============================================================================
# 配置
# ============================================================================

@dataclass
class RLPhysicalConfig:
    """RL 物理控制配置"""
    # 串口
    serial_port: str = "/dev/ttyACM1"
    serial_baud: int = 1_000_000
    serial_timeout: float = 0.02
    
    # RL 环境
    target_y: float = 0.5  # 目标 Y 位置
    target_z: float = 0.5  # 目标 Z 位置
    target_radius: float = 0.15  # 到达判定半径
    step_size: float = 0.05  # 每步位移
    max_steps: int = 50  # 最大步数
    
    # 控制
    action_delay: float = 0.3  # 动作间延时
    joint_step_rad: float = 0.05  # 关节每步弧度


# ============================================================================
# RL 物理环境
# ============================================================================

class RLPhysicalEnv:
    """
    RL 物理机械臂环境
    
    将 RL Agent 的动作映射到物理机械臂，
    同时维护虚拟位置状态用于 RL 决策。
    """
    
    def __init__(
        self,
        config: RLPhysicalConfig,
        serial_env: SerialArmEnv,
    ):
        self.cfg = config
        self.serial_env = serial_env
        
        # 虚拟位置（用于 RL 状态）
        self._y = 0.0
        self._z = 0.0
        self._target_y = config.target_y
        self._target_z = config.target_z
        self._step_count = 0
        
        # 状态维度
        self.observation_dim = 5  # [y, z, target_y, target_z, distance]
        self.action_dim = 4  # left, right, up, down
    
    def reset(self, target_y: Optional[float] = None, target_z: Optional[float] = None):
        """重置环境"""
        self._y = 0.0
        self._z = 0.0
        self._step_count = 0
        
        if target_y is not None:
            self._target_y = target_y
        if target_z is not None:
            self._target_z = target_z
        
        # 重置物理机械臂
        self.serial_env.reset()
        
        return self._get_obs()
    
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
            action: 0=left, 1=right, 2=up, 3=down
        """
        prev_dist = self._distance_to_target()
        
        # 更新虚拟位置
        dy = self.cfg.step_size if action == 0 else (-self.cfg.step_size if action == 1 else 0.0)
        dz = self.cfg.step_size if action == 2 else (-self.cfg.step_size if action == 3 else 0.0)
        
        self._y = np.clip(self._y + dy, -1.0, 1.0)
        self._z = np.clip(self._z + dz, -1.0, 1.0)
        self._step_count += 1
        
        # 发送到物理机械臂
        self.serial_env.step(action)
        
        # 计算奖励
        curr_dist = self._distance_to_target()
        reward = (prev_dist - curr_dist) * 1.0  # 靠近奖励
        reward -= 0.01  # 步数惩罚
        
        # 检查是否到达
        reached = curr_dist < self.cfg.target_radius
        if reached:
            reward += 10.0
        
        done = reached or (self._step_count >= self.cfg.max_steps)
        
        info = {
            "reached": reached,
            "distance": curr_dist,
            "steps": self._step_count,
            "position": (self._y, self._z),
            "target": (self._target_y, self._target_z),
        }
        
        return self._get_obs(), reward, done, info


# ============================================================================
# EEG 分类器
# ============================================================================

def load_eeg_classifier(dataset_type: str, subject_id: int, device: torch.device):
    """加载 CTNet EEG 分类器"""
    from CTNet_model import (
        EEGTransformer,
        BranchEEGNetTransformer,
        PatchEmbeddingCNN,
        PositioinalEncoding,
        TransformerEncoder,
        TransformerEncoderBlock,
        MultiHeadAttention,
        FeedForwardBlock,
        ResidualAdd,
        ClassificationHead,
    )
    
    # 数据集配置
    if dataset_type == "A":
        n_classes, n_channels = 4, 22
    else:
        n_classes, n_channels = 2, 3
    
    model = EEGTransformer(
        heads=2,
        emb_size=16,
        depth=6,
        database_type=dataset_type,
        eeg1_f1=8,
        eeg1_D=2,
        eeg1_kernel_size=64,
        eeg1_pooling_size1=8,
        eeg1_pooling_size2=8,
        eeg1_dropout_rate=0.5,
        eeg1_number_channel=n_channels,
        flatten_eeg1=240,
    ).to(device)
    
    # 尝试加载预训练权重
    model_path = _ROOT / "models" / "new" / f"model_{subject_id}.pth"
    if model_path.exists():
        # 注册安全类型（PyTorch 2.6+）
        safe_types = [
            EEGTransformer,
            BranchEEGNetTransformer,
            PatchEmbeddingCNN,
            PositioinalEncoding,
            TransformerEncoder,
            TransformerEncoderBlock,
            MultiHeadAttention,
            FeedForwardBlock,
            ResidualAdd,
            ClassificationHead,
        ]
        
        # 设置 __main__ 模块别名
        import sys
        main_mod = sys.modules.get("__main__")
        for cls in safe_types:
            setattr(main_mod, cls.__name__, cls)
        
        try:
            # 方法1: 使用 weights_only=False（信任本地模型）
            state = torch.load(model_path, map_location=device, weights_only=False)
            
            # 提取 state_dict
            if hasattr(state, 'state_dict'):
                state = state.state_dict()
            elif isinstance(state, dict):
                for key in ['state_dict', 'model_state', 'model', 'net', 'weights']:
                    if key in state and isinstance(state[key], dict):
                        state = state[key]
                        break
            
            model.load_state_dict(state)
            print(f"[CTNet] 已加载模型: {model_path}")
        except Exception as e:
            print(f"[CTNet] 加载模型失败: {e}")
    
    model.eval()
    return model


def load_rl_model(model_type: str, state_dim: int, action_dim: int, device: torch.device):
    """加载 RL 模型"""
    if model_type == "transformer":
        model = TransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1)
    elif model_type == "light_transformer":
        model = LightTransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1)
    else:
        model = DQNNetwork(state_dim=state_dim, action_dim=action_dim, seq_len=1)
    
    model = model.to(device)
    
    # 尝试加载预训练权重（优先使用 4 方向模型）
    model_paths = [
        _ROOT / "outputs" / "rl_4direction" / f"rl_4direction_{model_type}.pth",  # 4 方向模型
        _ROOT / "outputs" / "rl_control_test" / f"{model_type}_best.pth",
        _ROOT / "outputs" / f"dqn_{model_type}_best.pth",
        _ROOT / "outputs" / "dqn_policy_full.pth",
    ]
    
    for path in model_paths:
        if path.exists():
            try:
                # 使用 weights_only=False 信任本地模型
                state = torch.load(path, map_location=device, weights_only=False)
                
                # 提取 state_dict
                if hasattr(state, 'state_dict'):
                    state = state.state_dict()
                elif isinstance(state, dict) and 'state_dict' in state:
                    state = state['state_dict']
                
                model.load_state_dict(state)
                print(f"[RL] 已加载模型: {path}")
                break
            except Exception as e:
                print(f"[RL] 加载 {path} 失败: {e}")
    
    model.eval()
    return model


# ============================================================================
# 主控制循环
# ============================================================================

def run_rl_physical_control(
    config: RLPhysicalConfig,
    eeg_classifier: nn.Module,
    rl_model: nn.Module,
    eeg_data: torch.Tensor,
    eeg_labels: np.ndarray,
    device: torch.device,
    num_trials: int = 10,
    verbose: bool = True,
):
    """
    运行 RL 物理控制
    
    流程：
    1. 随机选择一个 EEG trial
    2. EEG 分类器预测意图
    3. 根据意图设定目标位置
    4. RL 模型控制机械臂到达目标
    """
    
    # 创建串口环境
    serial_cfg = SerialConfig(
        port=config.serial_port,
        baud=config.serial_baud,
        timeout=config.serial_timeout,
        joint_step_rad=config.joint_step_rad,
        max_steps=config.max_steps,
    )
    serial_env = SerialArmEnv(serial_cfg, render_mode="human")
    
    # 创建 RL 物理环境
    env = RLPhysicalEnv(config, serial_env)
    
    # 意图到目标位置的映射
    # 0=left, 1=right, 2=up/feet, 3=down/tongue (4类)
    # 0=left, 1=right (2类)
    n_classes = eeg_labels.max() + 1 if eeg_labels.min() == 0 else eeg_labels.max()
    
    if n_classes == 4:
        intent_to_target = {
            0: (-0.5, 0.0),   # left
            1: (0.5, 0.0),    # right
            2: (0.0, 0.5),    # up (feet)
            3: (0.0, -0.5),   # down (tongue)
        }
    else:
        intent_to_target = {
            0: (-0.5, 0.0),   # left
            1: (0.5, 0.0),    # right
        }
    
    action_names = ["left", "right", "up", "down"]
    results = []
    
    print("\n" + "="*60)
    print("RL 物理机械臂控制")
    print("="*60)
    print(f"设备: {config.serial_port}")
    print(f"目标半径: {config.target_radius}")
    print(f"最大步数: {config.max_steps}")
    print("="*60)
    
    for trial in range(num_trials):
        # 随机选择 EEG trial
        idx = np.random.randint(len(eeg_data))
        eeg_sample = eeg_data[idx:idx+1].to(device)
        true_label = int(eeg_labels[idx])
        if true_label >= 1 and eeg_labels.min() >= 1:
            true_label -= 1  # 0-based
        
        # EEG 分类预测
        eeg_classifier.eval()
        with torch.no_grad():
            logits = eeg_classifier(eeg_sample)
            if isinstance(logits, tuple):
                _, logits = logits
            pred_intent = logits.argmax(dim=1).item()
        
        # 设定目标
        target_y, target_z = intent_to_target.get(pred_intent, (0.5, 0.5))
        
        if verbose:
            print(f"\n[Trial {trial+1}/{num_trials}]")
            print(f"  EEG 预测意图: {action_names[pred_intent]} (真实: {action_names[true_label]})")
            print(f"  目标位置: ({target_y:.2f}, {target_z:.2f})")
        
        # 重置环境
        state = env.reset(target_y=target_y, target_z=target_z)
        
        total_reward = 0
        trajectory = [(0.0, 0.0)]
        actions_taken = []
        
        # RL 控制循环
        for step in range(config.max_steps):
            # RL 模型预测动作
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            rl_model.eval()
            with torch.no_grad():
                q_values = rl_model(state_tensor)
                action = q_values.argmax(dim=1).item()
            
            # 执行动作
            next_state, reward, done, info = env.step(action)
            
            total_reward += reward
            trajectory.append((env._y, env._z))
            actions_taken.append(action)
            
            if verbose and step % 5 == 0:
                print(f"    Step {step}: action={action_names[action]}, pos=({env._y:.2f}, {env._z:.2f}), dist={info['distance']:.3f}")
            
            # 动作延时
            time.sleep(config.action_delay)
            
            state = next_state
            
            if done:
                break
        
        # 记录结果（转换为 Python 原生类型以便 JSON 序列化）
        result = {
            "trial": int(trial + 1),
            "pred_intent": int(pred_intent),
            "true_label": int(true_label),
            "correct": bool(pred_intent == true_label),
            "reached": bool(info["reached"]),
            "steps": int(info["steps"]),
            "final_distance": float(info["distance"]),
            "total_reward": float(total_reward),
        }
        results.append(result)
        
        if verbose:
            status = "✓ 到达" if info["reached"] else "✗ 未到达"
            print(f"  结果: {status}, 步数: {info['steps']}, 奖励: {total_reward:.2f}")
    
    # 统计
    print("\n" + "="*60)
    print("统计结果")
    print("="*60)
    
    n_correct = sum(r["correct"] for r in results)
    n_reached = sum(r["reached"] for r in results)
    avg_steps = np.mean([r["steps"] for r in results])
    avg_reward = np.mean([r["total_reward"] for r in results])
    
    print(f"分类准确率: {n_correct}/{num_trials} ({100*n_correct/num_trials:.1f}%)")
    print(f"控制到达率: {n_reached}/{num_trials} ({100*n_reached/num_trials:.1f}%)")
    print(f"平均步数: {avg_steps:.1f}")
    print(f"平均奖励: {avg_reward:.2f}")
    print("="*60)
    
    # 关闭环境
    serial_env.close()
    
    return results


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="RL 模型控制物理机械臂")
    
    # 串口
    p.add_argument("--serial-port", type=str, required=True, help="串口端口")
    p.add_argument("--serial-baud", type=int, default=1_000_000)
    
    # 数据
    p.add_argument("--subject", type=int, default=1)
    p.add_argument("--dataset", choices=["A", "B"], default="A")
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    
    # 控制
    p.add_argument("--num-trials", type=int, default=10)
    p.add_argument("--action-delay", type=float, default=0.3, help="动作间延时（秒）")
    p.add_argument("--max-steps", type=int, default=30, help="每个 trial 最大步数")
    p.add_argument("--target-radius", type=float, default=0.15, help="到达判定半径")
    
    # 模型
    p.add_argument("--rl-model", choices=["dqn", "transformer", "light_transformer"], 
                   default="transformer")
    p.add_argument("--device", type=str, default="cuda")
    
    # 归位
    p.add_argument("--pre-home", action="store_true", help="启动前归位")
    p.add_argument("--post-home", action="store_true", help="结束后归位")
    p.add_argument("--home-json", type=Path, default=Path("serial_home.json"))
    p.add_argument("--return-json", type=Path, default=Path("serial_return.json"))
    
    return p.parse_args()


def main():
    args = parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    
    # 预归位
    if args.pre_home and args.home_json.exists():
        print("[Pre-home] 回到中位...")
        os.system(f"python scripts/serial_go_home.py --port {args.serial_port} --home-json {args.home_json}")
        time.sleep(1.0)
    
    # 加载数据
    from scipy.io import loadmat
    
    if args.dataset == "A":
        n_classes, n_channels = 4, 22
    else:
        n_classes, n_channels = 2, 3
    
    # 加载测试数据
    test_path = args.data_dir / f"{args.dataset}{args.subject:02d}E.mat"
    if not test_path.exists():
        test_path = args.data_dir / f"{args.dataset}{args.subject:02d}T.mat"
    
    data_mat = loadmat(test_path)
    eeg_data = data_mat["data"]
    eeg_labels = data_mat["label"].flatten()
    
    # 预处理
    eeg_data = np.expand_dims(eeg_data, axis=1)  # [N, 1, C, T]
    mean, std = eeg_data.mean(), eeg_data.std() or 1.0
    eeg_data = (eeg_data - mean) / std
    eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
    
    print(f"[Data] 加载 {len(eeg_tensor)} 个 EEG trials")
    
    # 加载模型
    eeg_classifier = load_eeg_classifier(args.dataset, args.subject, device)
    
    state_dim = 5  # [y, z, target_y, target_z, distance]
    action_dim = 4
    rl_model = load_rl_model(args.rl_model, state_dim, action_dim, device)
    
    # 配置
    config = RLPhysicalConfig(
        serial_port=args.serial_port,
        serial_baud=args.serial_baud,
        target_radius=args.target_radius,
        max_steps=args.max_steps,
        action_delay=args.action_delay,
    )
    
    # 运行控制
    try:
        results = run_rl_physical_control(
            config=config,
            eeg_classifier=eeg_classifier,
            rl_model=rl_model,
            eeg_data=eeg_tensor,
            eeg_labels=eeg_labels,
            device=device,
            num_trials=args.num_trials,
            verbose=True,
        )
        
        # 保存结果
        output_dir = _ROOT / "outputs" / "rl_physical_control"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n结果已保存: {output_dir / 'results.json'}")
        
    finally:
        # 结束归位
        if args.post_home and args.return_json.exists():
            print("\n[Post-home] 回到归位...")
            os.system(f"python scripts/serial_go_return.py --port {args.serial_port} --return-json {args.return_json}")


if __name__ == "__main__":
    main()

