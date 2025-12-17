"""DQN 网络骨架：1D-CNN + LSTM 特征提取 + 全连接 Q 头。

提供用于强化学习训练的核心模块：
    - DQNNetwork: 前向结构（输入 state -> Q-values）
    - ReplayBuffer: 经验回放存储
    - epsilon_greedy_action: 动作选择辅助函数
    - dqn_training_step: 单步训练逻辑（计算 loss、反向传播）

示例构造:
    model = DQNNetwork(state_dim=22, seq_len=10, action_dim=4)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """1D-CNN + LSTM 的 Q 网络骨架。

    输入张量形状: (batch, seq_len, state_dim)
        - seq_len: 允许堆叠多个时间步 (默认 1 表示单步)
        - state_dim: 单个时间步的特征维度（如 CSP 特征数量）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 1,
        conv_channels: int = 64,
        lstm_hidden: int = 128,
        fc_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len

        # 1D CNN 期望输入 (batch, channel, seq_len)，将特征视作“通道”
        self.conv = nn.Sequential(
            nn.Conv1d(state_dim, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # LSTM 接受 (batch, seq_len, features)，因此先转置
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, fc_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(fc_hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, state_dim)
        Returns:
            q_values: (batch, action_dim)
        """
        # 先把 state_dim 当作 channel
        x = x.transpose(1, 2)  # -> (batch, state_dim, seq_len)
        x = self.conv(x)
        x = x.transpose(1, 2)  # -> (batch, seq_len, conv_channels)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, lstm_hidden)
        last_hidden = lstm_out[:, -1, :]  # 仅取最后一个时间步
        q_values = self.fc(last_hidden)
        return q_values


@dataclass
class ReplayBuffer:
    """简单的经验回放缓冲区。"""

    capacity: int
    state_shape: Tuple[int, ...]
    device: torch.device

    def __post_init__(self) -> None:
        self.states = torch.zeros((self.capacity, *self.state_shape), dtype=torch.float32, device=self.device)
        self.actions = torch.zeros((self.capacity,), dtype=torch.long, device=self.device)
        self.rewards = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.next_states = torch.zeros((self.capacity, *self.state_shape), dtype=torch.float32, device=self.device)
        self.dones = torch.zeros((self.capacity,), dtype=torch.float32, device=self.device)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done) -> None:
        idx = self.ptr % self.capacity
        self.states[idx] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_states[idx] = torch.as_tensor(next_state, dtype=torch.float32, device=self.device)
        self.dones[idx] = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        idxs_t = torch.as_tensor(idxs, device=self.device)
        batch = (
            self.states[idxs_t],
            self.actions[idxs_t],
            self.rewards[idxs_t],
            self.next_states[idxs_t],
            self.dones[idxs_t],
        )
        return batch


def epsilon_greedy_action(q_values: torch.Tensor, epsilon: float) -> int:
    """按照 ε-greedy 策略选择动作。"""
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    return int(torch.argmax(q_values).item())


def dqn_training_step(
    policy_net: DQNNetwork,
    target_net: DQNNetwork,
    optimizer: torch.optim.Optimizer,
    batch,
    *,
    gamma: float = 0.99,
    huber_delta: float = 1.0,
) -> float:
    """单次参数更新，返回 loss 数值。"""
    states, actions, rewards, next_states, dones = batch

    q_values = policy_net(states)
    q_values = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target = rewards + (1 - dones) * gamma * next_q

    loss = F.smooth_l1_loss(q_values, target, reduction="mean")
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=huber_delta)
    optimizer.step()

    return loss.item()

