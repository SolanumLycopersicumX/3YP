#!/usr/bin/env python3
"""
Transformer-based DQN 网络

与 dqn_model.py 中的 CNN+LSTM 版本对比：
┌────────────────────────────────────────────────────────────────┐
│  原版 (DQNNetwork):     Conv1D → LSTM → FC → Q-values         │
│  本版 (TransformerDQN): PosEnc → Transformer → FC → Q-values  │
└────────────────────────────────────────────────────────────────┘

Transformer 优势：
1. 自注意力机制可以直接建模任意位置之间的关系
2. 并行计算效率高（相比 LSTM 的顺序处理）
3. 更好的长程依赖建模
4. 可扩展性强，适合大规模预训练

参考文献：
- Decision Transformer (Chen et al., 2021)
- Trajectory Transformer (Janner et al., 2021)

示例：
    model = TransformerDQN(state_dim=5, action_dim=4, seq_len=1)
    q_values = model(state)  # state: (batch, seq_len, state_dim)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 位置编码
# ============================================================================
class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码 (Vaswani et al., 2017)"""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x + positional_encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """可学习的位置编码"""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Transformer DQN 网络
# ============================================================================
class TransformerDQN(nn.Module):
    """
    Transformer-based Q 网络
    
    架构：
        Input Embedding → Positional Encoding → Transformer Encoder → Output Head
    
    输入张量形状: (batch, seq_len, state_dim)
    输出张量形状: (batch, action_dim)
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 1,
        d_model: int = 64,           # Transformer 隐藏维度
        n_heads: int = 4,            # 注意力头数
        n_layers: int = 2,           # Transformer 层数
        d_ff: int = 256,             # FFN 隐藏维度
        dropout: float = 0.1,
        pos_encoding: str = "sinusoidal",  # "sinusoidal" 或 "learnable"
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.d_model = d_model
        
        # 1. 输入投影层：state_dim → d_model
        self.input_projection = nn.Linear(state_dim, d_model)
        
        # 2. 位置编码
        if pos_encoding == "sinusoidal":
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        else:
            self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,  # 使用 (batch, seq, feature) 格式
            norm_first=True,   # Pre-LN 架构，训练更稳定
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )
        
        # 4. 输出头
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, state_dim)
        Returns:
            q_values: (batch, action_dim)
        """
        # 1. 输入投影
        x = self.input_projection(x)  # → (batch, seq_len, d_model)
        
        # 2. 添加位置编码
        x = self.pos_encoder(x)  # → (batch, seq_len, d_model)
        
        # 3. Transformer 编码
        x = self.transformer_encoder(x)  # → (batch, seq_len, d_model)
        
        # 4. 取最后一个时间步的输出（类似 LSTM 的做法）
        x = x[:, -1, :]  # → (batch, d_model)
        
        # 5. 输出 Q 值
        q_values = self.output_head(x)  # → (batch, action_dim)
        
        return q_values
    
    def get_attention_weights(self, x: torch.Tensor) -> list:
        """获取注意力权重（用于可视化）"""
        # 注意：需要修改 TransformerEncoder 才能提取注意力权重
        # 这里提供一个简化版本
        pass


# ============================================================================
# Decision Transformer 风格的网络（更高级版本）
# ============================================================================
class DecisionTransformerDQN(nn.Module):
    """
    Decision Transformer 风格的 Q 网络
    
    特点：
    1. 将 (state, action, reward) 作为一个 token 序列
    2. 可以处理历史轨迹信息
    3. 支持 Return-to-Go 条件生成
    
    这是一个简化版本，主要用于对比实验。
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_seq_len: int = 20,       # 最大序列长度
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Token 类型嵌入
        # 3 种 token: state, action, return-to-go
        self.token_type_emb = nn.Embedding(3, d_model)
        
        # 各类型的投影层
        self.state_encoder = nn.Linear(state_dim, d_model)
        self.action_encoder = nn.Embedding(action_dim, d_model)
        self.return_encoder = nn.Linear(1, d_model)
        
        # 位置编码
        self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=max_seq_len * 3)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 输出头：预测动作
        self.action_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, action_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        states: torch.Tensor,                    # (batch, seq_len, state_dim)
        actions: Optional[torch.Tensor] = None,  # (batch, seq_len) or None
        returns_to_go: Optional[torch.Tensor] = None,  # (batch, seq_len, 1) or None
    ) -> torch.Tensor:
        """
        Args:
            states: 状态序列
            actions: 动作序列（可选，用于条件生成）
            returns_to_go: 剩余回报（可选，用于条件生成）
        
        Returns:
            action_logits: (batch, action_dim) 动作 logits
        """
        batch_size, seq_len, _ = states.shape
        device = states.device
        
        # 编码状态
        state_emb = self.state_encoder(states)  # (batch, seq_len, d_model)
        state_emb = state_emb + self.token_type_emb(
            torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        )
        
        tokens = [state_emb]
        
        # 如果有动作历史，也编码
        if actions is not None:
            action_emb = self.action_encoder(actions)  # (batch, seq_len, d_model)
            action_emb = action_emb + self.token_type_emb(
                torch.ones(batch_size, seq_len, dtype=torch.long, device=device)
            )
            tokens.append(action_emb)
        
        # 如果有 return-to-go，也编码
        if returns_to_go is not None:
            rtg_emb = self.return_encoder(returns_to_go)  # (batch, seq_len, d_model)
            rtg_emb = rtg_emb + self.token_type_emb(
                torch.full((batch_size, seq_len), 2, dtype=torch.long, device=device)
            )
            tokens.append(rtg_emb)
        
        # 交错排列 tokens: [s1, a1, r1, s2, a2, r2, ...]
        # 简化版：直接拼接
        x = torch.cat(tokens, dim=1)  # (batch, seq_len * n_token_types, d_model)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # 取最后一个 state token 的输出来预测动作
        # 对于简化版，取最后一个 token
        x = x[:, -1, :]  # (batch, d_model)
        
        # 输出动作 logits
        action_logits = self.action_head(x)  # (batch, action_dim)
        
        return action_logits


# ============================================================================
# 轻量级 Transformer（适合快速实验）
# ============================================================================
class LightTransformerDQN(nn.Module):
    """
    轻量级 Transformer Q 网络
    
    特点：
    1. 参数量小，训练快
    2. 单层 Transformer + 残差连接
    3. 适合快速对比实验
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        seq_len: int = 1,
        d_model: int = 32,
        n_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.state_dim = state_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_proj = nn.Linear(state_dim, d_model)
        
        # 单层自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # 输出头
        self.output_head = nn.Linear(d_model, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, state_dim)
        Returns:
            q_values: (batch, action_dim)
        """
        # 输入投影
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        
        # Self-Attention + 残差
        attn_out, _ = self.self_attn(x, x, x)
        x = self.attn_norm(x + attn_out)
        
        # FFN + 残差
        ffn_out = self.ffn(x)
        x = self.ffn_norm(x + ffn_out)
        
        # 取最后时间步
        x = x[:, -1, :]  # (batch, d_model)
        
        # 输出 Q 值
        q_values = self.output_head(x)
        
        return q_values


# ============================================================================
# 工厂函数：统一接口创建不同网络
# ============================================================================
def create_dqn_network(
    network_type: str,
    state_dim: int,
    action_dim: int,
    seq_len: int = 1,
    **kwargs,
) -> nn.Module:
    """
    创建 DQN 网络的工厂函数
    
    Args:
        network_type: "cnn_lstm" | "transformer" | "light_transformer" | "decision_transformer"
        state_dim: 状态维度
        action_dim: 动作维度
        seq_len: 序列长度
        **kwargs: 网络特定参数
    
    Returns:
        nn.Module: DQN 网络实例
    """
    if network_type == "cnn_lstm":
        # 导入原始版本
        from scripts.dqn_model import DQNNetwork
        return DQNNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            conv_channels=kwargs.get("conv_channels", 64),
            lstm_hidden=kwargs.get("lstm_hidden", 128),
            fc_hidden=kwargs.get("fc_hidden", 128),
        )
    
    elif network_type == "transformer":
        return TransformerDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            d_model=kwargs.get("d_model", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            d_ff=kwargs.get("d_ff", 256),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    elif network_type == "light_transformer":
        return LightTransformerDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            seq_len=seq_len,
            d_model=kwargs.get("d_model", 32),
            n_heads=kwargs.get("n_heads", 2),
            dropout=kwargs.get("dropout", 0.1),
        )
    
    elif network_type == "decision_transformer":
        return DecisionTransformerDQN(
            state_dim=state_dim,
            action_dim=action_dim,
            max_seq_len=kwargs.get("max_seq_len", 20),
            d_model=kwargs.get("d_model", 128),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 3),
        )
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# ============================================================================
# 模型参数统计
# ============================================================================
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compare_networks(state_dim: int = 5, action_dim: int = 4, seq_len: int = 1):
    """打印不同网络的参数量对比"""
    try:
        from scripts.dqn_model import DQNNetwork
    except ModuleNotFoundError:
        from dqn_model import DQNNetwork
    
    networks = {
        "CNN+LSTM (原版)": DQNNetwork(state_dim, action_dim, seq_len),
        "LightTransformer": LightTransformerDQN(state_dim, action_dim, seq_len),
        "Transformer": TransformerDQN(state_dim, action_dim, seq_len),
        "DecisionTransformer": DecisionTransformerDQN(state_dim, action_dim),
    }
    
    print("=" * 60)
    print("网络参数量对比")
    print("=" * 60)
    print(f"{'网络类型':<25} {'总参数':>12} {'可训练参数':>12}")
    print("-" * 60)
    
    for name, model in networks.items():
        total, trainable = count_parameters(model)
        print(f"{name:<25} {total:>12,} {trainable:>12,}")
    
    print("=" * 60)


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("测试 Transformer DQN 网络\n")
    
    # 参数
    batch_size = 8
    seq_len = 1
    state_dim = 5
    action_dim = 4
    
    # 测试输入
    x = torch.randn(batch_size, seq_len, state_dim)
    
    # 测试各个网络
    print("1. TransformerDQN")
    model1 = TransformerDQN(state_dim, action_dim, seq_len)
    out1 = model1(x)
    print(f"   输入: {x.shape} → 输出: {out1.shape}")
    total, _ = count_parameters(model1)
    print(f"   参数量: {total:,}\n")
    
    print("2. LightTransformerDQN")
    model2 = LightTransformerDQN(state_dim, action_dim, seq_len)
    out2 = model2(x)
    print(f"   输入: {x.shape} → 输出: {out2.shape}")
    total, _ = count_parameters(model2)
    print(f"   参数量: {total:,}\n")
    
    print("3. DecisionTransformerDQN")
    model3 = DecisionTransformerDQN(state_dim, action_dim)
    out3 = model3(x)
    print(f"   输入: {x.shape} → 输出: {out3.shape}")
    total, _ = count_parameters(model3)
    print(f"   参数量: {total:,}\n")
    
    # 对比所有网络
    compare_networks(state_dim, action_dim, seq_len)

