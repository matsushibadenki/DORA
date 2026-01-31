# snn_research/models/transformer/logic_gated_attention.py
# Title: Logic-Gated Spiking Self-Attention (Fix Imports)

import torch
import torch.nn as nn
from typing import Optional, Tuple
from spikingjelly.activation_based import layer

from snn_research.core.neurons.da_lif_node import DualAdaptiveLIFNode

# [Fix] Use existing logic_gated_snn module
from snn_research.core.layers.logic_gated_snn import PhaseCriticalSCAL as SCALPerceptionLayer

class LogicGateController(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        self.logic_core = SCALPerceptionLayer(
            in_features=dim,
            out_features=num_heads,
            time_steps=1,
            v_th_init=0.5,
            gain=10.0
        )
        
        self.proj_q = nn.Linear(dim, dim // 2)
        self.proj_k = nn.Linear(dim, dim // 2)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, H, N, D = q.shape
        energy = (q * k).sum(dim=-1).mean(dim=-1) # (B, H)
        energy_feat = energy.repeat(1, self.dim // H) # (B, Dim)
        
        gate_signals = self.logic_core(energy_feat)['output'] # (B, H)
        gate = gate_signals.view(B, H, 1, 1)
        
        return gate


class LogicGatedSpikingSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, tau_m: float = 2.0):
        super().__init__()
        assert d_model % num_heads == 0
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

        self.logic_gate = LogicGateController(d_model, num_heads)

        self.attn_lif = DualAdaptiveLIFNode(
            tau_m_init=tau_m, v_threshold=0.5, detach_reset=True)

        self.proj_linear = layer.Linear(d_model, d_model)
        self.proj_bn = nn.BatchNorm1d(d_model)
        self.proj_lif = DualAdaptiveLIFNode(tau_m_init=tau_m, detach_reset=True)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape

        q = self.q_lif(self.q_bn(self.q_linear(x).transpose(1, 2)).transpose(1, 2))
        k = self.k_lif(self.k_bn(self.k_linear(x).transpose(1, 2)).transpose(1, 2))
        v = self.v_lif(self.v_bn(self.v_linear(x).transpose(1, 2)).transpose(1, 2))

        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        logic_mask = self.logic_gate(q, k)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn_gated = attn_score * logic_mask

        x_attn = attn_gated @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, D)

        x_attn = self.attn_lif(x_attn)
        x_attn = self.proj_lif(self.proj_bn(self.proj_linear(x_attn).transpose(1, 2)).transpose(1, 2))

        return x_attn