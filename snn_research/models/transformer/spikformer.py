# snn_research/models/transformer/spikformer.py
# Title: Spikformer (Phase 2 Optimized)
# Description:
#   Auto-Tuned Configuration Support.
#   Flash Attention & Batch-Time Merging optimized for MPS/CUDA.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union
from spikingjelly.activation_based import functional as SJ_F
from spikingjelly.activation_based import layer

from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode
from snn_research.core.base import BaseModel


class SpikingSelfAttention(nn.Module):
    """
    Softmax-less SSA with Flash Attention Optimization.
    T次元を展開せず、(B*T)で一括処理することで高速化を実現。
    """

    def __init__(self, d_model: int, num_heads: int, tau_m: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.125

        self.q_linear = layer.Linear(d_model, d_model)
        self.q_bn = nn.BatchNorm1d(d_model)
        self.q_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.k_linear = layer.Linear(d_model, d_model)
        self.k_bn = nn.BatchNorm1d(d_model)
        self.k_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.v_linear = layer.Linear(d_model, d_model)
        self.v_bn = nn.BatchNorm1d(d_model)
        self.v_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.attn_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(d_model, d_model)
        self.proj_bn = nn.BatchNorm1d(d_model)
        self.proj_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        # x: (B*T, N, D) - Batch and Time are merged
        
        # 1. Linear Projections (Batch*Time 一括処理)
        q = self.q_linear(x)
        q = self.q_lif(self.q_bn(q.transpose(1, 2)).transpose(1, 2))
        
        k = self.k_linear(x)
        k = self.k_lif(self.k_bn(k.transpose(1, 2)).transpose(1, 2))
        
        v = self.v_linear(x)
        v = self.v_lif(self.v_bn(v.transpose(1, 2)).transpose(1, 2))

        # 2. Reshape for Multi-head Attention
        # (B*T, N, D) -> (B*T, N, Num_Heads, Head_Dim) -> (B*T, Num_Heads, N, Head_Dim)
        B_T, N, D = x.shape
        q = q.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B_T, N, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Spiking Self Attention (SSA)
        # Scaled Dot Product
        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        
        # SNN特有: Softmaxを使わず直接Vを掛ける (Spikformer仕様)
        x_attn = attn_score @ v

        # 4. Output Projection
        x_attn = x_attn.transpose(1, 2).reshape(B_T, N, D)
        x_attn = self.attn_lif(x_attn)
        
        x_attn = self.proj_linear(x_attn)
        x_attn = self.proj_lif(self.proj_bn(x_attn.transpose(1, 2)).transpose(1, 2))

        return x_attn


class SpikingMLP(nn.Module):
    def __init__(self, d_model: int, mlp_ratio: int = 4, tau_m: float = 2.0):
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.fc1 = layer.Linear(d_model, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.lif1 = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

        self.fc2 = layer.Linear(hidden_dim, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.lif2 = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        # x: (B*T, N, D)
        x = self.fc1(x)
        x = self.lif1(self.bn1(x.transpose(1, 2)).transpose(1, 2))
        
        x = self.fc2(x)
        x = self.lif2(self.bn2(x.transpose(1, 2)).transpose(1, 2))
        return x


class SpikformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = SpikingSelfAttention(d_model, num_heads)
        self.mlp = SpikingMLP(d_model, mlp_ratio)

    def forward(self, x: torch.Tensor):
        # Residual Connection
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class Spikformer(BaseModel):
    def __init__(
        self,
        img_size_h: int = 224,
        img_size_w: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        mlp_ratio: int = 4,
        T: int = 4,
        num_classes: int = 1000
    ):
        super().__init__()
        self.T = T
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        # Patch Embedding
        self.patch_embed = layer.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.bn_embed = nn.BatchNorm2d(embed_dim)
        self.lif_embed = DualAdaptiveLIFNode(detach_reset=True)

        num_patches = (img_size_h // patch_size) * (img_size_w // patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList([
            SpikformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(num_layers)
        ])

        self.head = layer.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x: torch.Tensor):
        # x: (B*T, C, H, W) -> Batch and Time merged
        
        # Patch Embedding
        x = self.patch_embed(x)
        x = self.lif_embed(self.bn_embed(x))
        x = x.flatten(2).transpose(1, 2) # (B*T, N, D)
        
        # Add Positional Embedding (Broadcasting works for B*T)
        x = x + self.pos_embed
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor):
        """
        Optimized forward pass using Batch-Time Merging.
        T=1 の場合は余計な次元拡張を行わずに最速パスを通す。
        """
        
        # Case 1: T=1 Optimized Path (Pure Spatial)
        if self.T == 1:
            if x.dim() == 5: # (B, T, C, H, W) -> (B, C, H, W)
                x = x.squeeze(1)
            
            # Reset not strictly needed for T=1 stateless, but good for safety
            SJ_F.reset_net(self)
            
            features = self.forward_features(x) # (B, N, D)
            x_gap = features.mean(dim=1)        # (B, D)
            return self.head(x_gap)

        # Case 2: Temporal Processing with Batch-Time Merging
        else:
            # 入力形状の正規化
            if x.dim() == 4: # (B, C, H, W) -> (B, T, C, H, W)
                x = x.unsqueeze(1).repeat(1, self.T, 1, 1, 1)
            
            B, T, C, H, W = x.shape
            
            # ニューロンの状態リセット
            SJ_F.reset_net(self)

            # Batch-Time Merging: (B*T, C, H, W)
            x_flatten = x.reshape(B * T, C, H, W)
            
            features = self.forward_features(x_flatten) # (B*T, N, D)
            
            # 時間方向の集約
            features = features.view(B, T, -1, self.embed_dim) # (B, T, N, D)
            x_mean_time = features.mean(dim=1) # (B, N, D)
            
            # トークン方向の集約
            x_gap = x_mean_time.mean(dim=1) # (B, D)
            
            return self.head(x_gap)


class TransformerToMambaAdapter(nn.Module):
    def __init__(self, vis_dim: int, model_dim: int, seq_len: Optional[int] = None):
        super().__init__()
        self.proj = nn.Linear(vis_dim, model_dim)
        self.ln = nn.LayerNorm(model_dim)

    def forward(self, x_vis: torch.Tensor) -> torch.Tensor:
        if x_vis.dim() == 4:
            x_vis = x_vis.mean(dim=1)
        return self.ln(self.proj(x_vis))