#!/usr/bin/env python3
"""
三数据集 RL 控制测试

对 IV-2a, IV-2b, GigaScience 三个数据集进行 RL 控制测试

流程:
1. 用 CTNet 进行 EEG 分类
2. 将分类结果作为 RL Agent 的输入
3. RL Agent 控制 2D 环境中的机械臂
4. 评估控制性能 (到达率、轨迹平滑度等)

创建时间: 2026-02-10
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn

# 项目路径设置
import sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.dqn_model import DQNNetwork, ReplayBuffer, epsilon_greedy_action
from scripts.dqn_transformer import TransformerDQN, LightTransformerDQN
from scripts.train_dqn_rl import RLArm2DEnv, RLArmConfig


# ============================================================================
# 数据集配置
# ============================================================================

@dataclass
class DatasetConfig:
    """数据集配置"""
    name: str
    type_code: str  # A, B, G
    n_channels: int
    n_classes: int
    n_subjects: int
    random_baseline: float
    action_mapping: Dict[int, str] = field(default_factory=dict)

DATASETS = {
    'IV-2a': DatasetConfig(
        name='BCI Competition IV-2a',
        type_code='A',
        n_channels=22,
        n_classes=4,
        n_subjects=9,
        random_baseline=0.25,
        action_mapping={0: 'left', 1: 'right', 2: 'down', 3: 'up'}
    ),
    'IV-2b': DatasetConfig(
        name='BCI Competition IV-2b',
        type_code='B',
        n_channels=3,
        n_classes=2,
        n_subjects=9,
        random_baseline=0.50,
        action_mapping={0: 'left', 1: 'right'}
    ),
    'GigaScience': DatasetConfig(
        name='GigaScience (Jeong 2020)',
        type_code='G',
        n_channels=64,
        n_classes=11,
        n_subjects=25,
        random_baseline=1/11,
        action_mapping={
            0: 'supination', 1: 'pronation', 2: 'wrist_ext', 3: 'wrist_flex',
            4: 'finger_ext', 5: 'fist', 6: 'lateral_grasp', 7: 'palmar_grasp',
            8: 'elbow_flex', 9: 'elbow_ext', 10: 'rest'
        }
    ),
}


# ============================================================================
# EEG 分类器 (使用预训练的 CTNet 或简化版本)
# ============================================================================

class SimpleEEGClassifier(nn.Module):
    """简化的 EEG 分类器 (用于 RL 控制测试)"""
    
    def __init__(self, n_channels: int, n_classes: int, seq_len: int = 1000):
        super().__init__()
        
        # 简化的 CNN 特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        x = self.conv(x)
        x = self.classifier(x)
        return x


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_channels: int,
    n_classes: int,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> SimpleEEGClassifier:
    """训练简化的 EEG 分类器"""
    
    # 准备数据
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train.flatten() - 1, dtype=torch.long)  # 标签从 0 开始
    
    # 归一化
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)
    
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    seq_len = X.shape[-1]
    model = SimpleEEGClassifier(n_channels, n_classes, seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    
    return model


def evaluate_classifier(
    model: SimpleEEGClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Tuple[float, np.ndarray]:
    """评估分类器并返回预测结果"""
    
    X = torch.tensor(X_test, dtype=torch.float32)
    y = torch.tensor(y_test.flatten() - 1, dtype=torch.long)
    
    # 使用训练集的归一化参数
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-8)
    
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        out = model(X)
        preds = out.argmax(dim=1).cpu().numpy()
        accuracy = (preds == y.numpy()).mean()
    
    return accuracy, preds


# ============================================================================
# RL 控制测试
# ============================================================================

@dataclass
class RLControlResult:
    """RL 控制测试结果"""
    classification_accuracy: float
    control_reach_rate: float
    avg_steps: float
    avg_reward: float
    trajectory_smoothness: float
    
    def to_dict(self) -> Dict:
        return {
            'classification_accuracy': self.classification_accuracy,
            'control_reach_rate': self.control_reach_rate,
            'avg_steps': self.avg_steps,
            'avg_reward': self.avg_reward,
            'trajectory_smoothness': self.trajectory_smoothness,
        }


def run_rl_control_episode(
    env: RLArm2DEnv,
    rl_model: nn.Module,
    eeg_predictions: np.ndarray,
    device: torch.device,
    max_steps: int = 100,
) -> Dict[str, Any]:
    """
    运行一个 RL 控制 episode
    
    使用 EEG 预测作为 RL 状态的一部分
    """
    state = env.reset()
    
    total_reward = 0
    steps = 0
    positions = [np.array([env._y, env._z])]  # 使用内部属性
    actions = []
    
    # 随机选择一个 EEG 预测作为当前意图
    trial_idx = np.random.randint(len(eeg_predictions))
    intended_action = eeg_predictions[trial_idx]
    
    for step in range(max_steps):
        # 构建 RL 状态: [position, target, intended_action_onehot]
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        rl_model.eval()
        with torch.no_grad():
            q_values = rl_model(state_tensor)
            action = q_values.argmax(dim=1).item()
        
        next_state, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        positions.append(np.array([env._y, env._z]))
        actions.append(action)
        
        state = next_state
        
        if done:
            break
    
    # 计算轨迹平滑度 (动作变化次数)
    if len(actions) > 1:
        action_changes = sum(1 for i in range(1, len(actions)) if actions[i] != actions[i-1])
        smoothness = 1 - (action_changes / len(actions))
    else:
        smoothness = 1.0
    
    return {
        'reached': info.get('reached', False),
        'steps': steps,
        'reward': total_reward,
        'smoothness': smoothness,
        'positions': positions,
    }


def run_rl_control_test(
    dataset_config: DatasetConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    rl_model_type: str = "transformer",
    n_episodes: int = 100,
    classifier_epochs: int = 50,
    rl_training_episodes: int = 500,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> RLControlResult:
    """
    对单个数据集运行完整的 RL 控制测试
    """
    n_channels = dataset_config.n_channels
    n_classes = dataset_config.n_classes
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"RL 控制测试: {dataset_config.name}")
        print(f"{'='*60}")
        print(f"通道: {n_channels}, 类别: {n_classes}")
    
    # 1. 训练 EEG 分类器
    if verbose:
        print("\n[Step 1] 训练 EEG 分类器...")
    
    classifier = train_classifier(
        X_train, y_train, n_channels, n_classes,
        epochs=classifier_epochs, device=device
    )
    
    # 2. 评估分类器
    if verbose:
        print("[Step 2] 评估分类器...")
    
    cls_accuracy, predictions = evaluate_classifier(classifier, X_test, y_test, device)
    
    if verbose:
        print(f"  分类准确率: {cls_accuracy*100:.2f}%")
    
    # 3. 训练 RL Agent
    if verbose:
        print("[Step 3] 训练 RL Agent...")
    
    env_config = RLArmConfig(max_steps=100, target_radius=0.1)
    env = RLArm2DEnv(env_config)
    
    # 选择 RL 模型
    state_dim = env.observation_dim
    action_dim = 4  # 上下左右
    
    if rl_model_type == "transformer":
        rl_model = TransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    elif rl_model_type == "light_transformer":
        rl_model = LightTransformerDQN(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    else:
        rl_model = DQNNetwork(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    
    target_net = type(rl_model)(state_dim=state_dim, action_dim=action_dim, seq_len=1).to(device)
    target_net.load_state_dict(rl_model.state_dict())
    
    optimizer = torch.optim.Adam(rl_model.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=10000, state_shape=(1, state_dim), device=device)
    
    # 简化的 RL 训练
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    gamma = 0.99
    batch_size = 64
    
    for ep in range(rl_training_episodes):
        state = env.reset()
        episode_reward = 0
        
        for step in range(100):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            if np.random.random() < epsilon:
                action = np.random.randint(action_dim)
            else:
                rl_model.eval()
                with torch.no_grad():
                    q_values = rl_model(state_tensor)
                    action = q_values.argmax(dim=1).item()
            
            next_state, reward, done, _ = env.step(action)
            
            replay_buffer.add(
                state.reshape(1, -1),
                action,
                reward,
                next_state.reshape(1, -1),
                done
            )
            
            episode_reward += reward
            state = next_state
            
            # 训练
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                rl_model.train()
                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1)[0]
                    target_q = rewards + gamma * next_q * (1 - dones)
                
                current_q = rl_model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                loss = nn.functional.smooth_l1_loss(current_q, target_q)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if done:
                break
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
        # 软更新 target network
        if ep % 10 == 0:
            for param, target_param in zip(rl_model.parameters(), target_net.parameters()):
                target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)
    
    if verbose:
        print(f"  RL 训练完成 ({rl_training_episodes} episodes)")
    
    # 4. 控制测试
    if verbose:
        print("[Step 4] 控制测试...")
    
    results = []
    for ep in range(n_episodes):
        result = run_rl_control_episode(env, rl_model, predictions, device)
        results.append(result)
    
    reach_rate = sum(r['reached'] for r in results) / len(results)
    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_smoothness = np.mean([r['smoothness'] for r in results])
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"结果总结: {dataset_config.name}")
        print(f"{'='*60}")
        print(f"  分类准确率:   {cls_accuracy*100:.2f}%")
        print(f"  控制到达率:   {reach_rate*100:.2f}%")
        print(f"  平均步数:     {avg_steps:.1f}")
        print(f"  平均奖励:     {avg_reward:.2f}")
        print(f"  轨迹平滑度:   {avg_smoothness:.3f}")
    
    return RLControlResult(
        classification_accuracy=cls_accuracy,
        control_reach_rate=reach_rate,
        avg_steps=avg_steps,
        avg_reward=avg_reward,
        trajectory_smoothness=avg_smoothness,
    )


# ============================================================================
# 三数据集对比测试
# ============================================================================

def run_three_dataset_comparison(
    data_dir: Path,
    output_dir: Path,
    subjects_per_dataset: Optional[Dict[str, List[int]]] = None,
    rl_model_type: str = "transformer",
    device: str = "cuda",
    verbose: bool = True,
) -> Dict[str, Dict]:
    """
    运行三个数据集的对比测试
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    if subjects_per_dataset is None:
        subjects_per_dataset = {
            'IV-2a': [1, 2, 3],  # 测试前 3 个被试
            'IV-2b': [1, 2, 3],
            'GigaScience': [1, 2, 3],
        }
    
    all_results = {}
    
    for dataset_name, config in DATASETS.items():
        subjects = subjects_per_dataset.get(dataset_name, [])
        
        if not subjects:
            print(f"\n跳过 {dataset_name} (无被试)")
            continue
        
        dataset_results = []
        
        for subject in subjects:
            try:
                # 加载数据
                if config.type_code in ['A', 'B']:
                    from utils import load_data_evaluate
                    X_train, y_train, X_test, y_test = load_data_evaluate(
                        str(data_dir) + "/", config.type_code, subject, "subject-dependent"
                    )
                else:
                    from scripts.gigascience_loader import load_gigascience_evaluate
                    X_train, y_train, X_test, y_test = load_gigascience_evaluate(
                        data_dir, subject, "subject-dependent"
                    )
                
                # 运行测试
                result = run_rl_control_test(
                    config, X_train, y_train, X_test, y_test,
                    rl_model_type=rl_model_type,
                    device=device,
                    verbose=verbose,
                )
                
                dataset_results.append({
                    'subject': subject,
                    **result.to_dict(),
                })
                
            except Exception as e:
                print(f"  {dataset_name} Subject {subject} 失败: {e}")
        
        if dataset_results:
            all_results[dataset_name] = {
                'subjects': dataset_results,
                'mean_classification': np.mean([r['classification_accuracy'] for r in dataset_results]),
                'mean_reach_rate': np.mean([r['control_reach_rate'] for r in dataset_results]),
                'mean_smoothness': np.mean([r['trajectory_smoothness'] for r in dataset_results]),
            }
    
    # 保存结果
    with open(output_dir / "rl_control_comparison.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 可视化
    visualize_rl_comparison(all_results, output_dir / "rl_control_comparison.png")
    
    return all_results


def visualize_rl_comparison(results: Dict, output_path: Path):
    """可视化三数据集 RL 控制对比"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    datasets = list(results.keys())
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    
    # 1. 分类准确率
    ax1 = axes[0]
    cls_acc = [results[d]['mean_classification'] * 100 for d in datasets]
    bars1 = ax1.bar(datasets, cls_acc, color=colors[:len(datasets)], alpha=0.8)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Classification Accuracy', fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, cls_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.1f}%', ha='center', fontsize=10)
    
    # 2. 控制到达率
    ax2 = axes[1]
    reach = [results[d]['mean_reach_rate'] * 100 for d in datasets]
    bars2 = ax2.bar(datasets, reach, color=colors[:len(datasets)], alpha=0.8)
    ax2.set_ylabel('Reach Rate (%)')
    ax2.set_title('RL Control Reach Rate', fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, r in zip(bars2, reach):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{r:.1f}%', ha='center', fontsize=10)
    
    # 3. 轨迹平滑度
    ax3 = axes[2]
    smooth = [results[d]['mean_smoothness'] for d in datasets]
    bars3 = ax3.bar(datasets, smooth, color=colors[:len(datasets)], alpha=0.8)
    ax3.set_ylabel('Smoothness')
    ax3.set_title('Trajectory Smoothness', fontweight='bold')
    ax3.set_ylim(0, 1)
    for bar, s in zip(bars3, smooth):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{s:.3f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n可视化已保存: {output_path}")


# ============================================================================
# 主程序
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="三数据集 RL 控制测试")
    
    p.add_argument("--data-dir", type=Path, default=Path("./mymat_raw/"))
    p.add_argument("--output-dir", type=Path, default=Path("./outputs/rl_control_test/"))
    p.add_argument("--model", choices=["dqn", "transformer", "light_transformer"],
                   default="transformer")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3],
                   help="每个数据集测试的被试")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("三数据集 RL 控制测试")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"RL 模型: {args.model}")
    print(f"设备: {args.device}")
    print("="*60)
    
    subjects_config = {
        'IV-2a': args.subjects,
        'IV-2b': args.subjects,
        # 'GigaScience': args.subjects,  # 暂时禁用，需要先下载数据
    }
    
    results = run_three_dataset_comparison(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subjects_per_dataset=subjects_config,
        rl_model_type=args.model,
        device=args.device,
    )
    
    # 打印最终总结
    print("\n" + "="*60)
    print("最终结果总结")
    print("="*60)
    print(f"{'数据集':<15} {'分类准确率':<15} {'控制到达率':<15} {'平滑度':<10}")
    print("-"*60)
    for dataset, res in results.items():
        print(f"{dataset:<15} {res['mean_classification']*100:>10.2f}%    "
              f"{res['mean_reach_rate']*100:>10.2f}%    {res['mean_smoothness']:>8.3f}")
    print("="*60)


if __name__ == "__main__":
    main()

