# snn_research/models/experimental/bit_spike_mamba.py
# Title: BitSpikeMamba v2.0 (SSM Integration)
# Description: 
#   ROADMAP Phase 2に基づき、BitNetとMamba(SSM)を融合したアーキテクチャを実装。
#   単純なFFNから、BitLinearを用いた簡易SSM(State Space Model)ブロックへアップグレード。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Optional


def bit_quantize_weight(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Weights to {-1, 0, 1} (1.58bit) quantization function"""
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_quant = (w_scaled).round().clamp(-1, 1)
    w_quant = (w_quant - w_scaled).detach() + w_scaled
    return w_quant, scale


class BitLinear(nn.Linear):
    """Linear layer with BitNet quantization"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant, scale = bit_quantize_weight(self.weight)
        return F.linear(x, w_quant) * scale + (self.bias if self.bias is not None else 0)


class BitSSM(nn.Module):
    """
    Simplified State Space Model Block using BitLinear.
    Mimics the selection mechanism of Mamba but optimized for low-bit computation.
    """
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # In-projection (BitLinear)
        self.in_proj = BitLinear(d_model, self.d_inner * 2)

        # Convolution (Simplified temporal mixing)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            groups=self.d_inner,
            padding=3
        )

        # State Space parameters (A, B, C) - approximate
        # x_proj maps input to B, C, delta
        self.x_proj = BitLinear(self.d_inner, self.d_state + d_model + self.d_inner)
        
        # Out-projection
        self.out_proj = BitLinear(self.d_inner, d_model)
        
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Seq, Dim)
        batch, seq, dim = x.shape
        
        # 1. Project to higher dimension
        x_and_res = self.in_proj(x)  # (B, L, 2*D_inner)
        x_inner, res = x_and_res.split(self.d_inner, dim=-1)
        
        # 2. Convolution (Temporal Mixing)
        # Transpose for Conv1d: (B, Dim, Seq)
        x_conv = x_inner.permute(0, 2, 1)
        x_conv = self.conv1d(x_conv)[:, :, :seq]
        x_conv = self.act(x_conv.permute(0, 2, 1)) # Back to (B, Seq, Dim)

        # 3. SSM Approximation (Simplified Gating)
        # In a full Mamba, this is the Selective Scan. 
        # Here we use a gated recurrence approximation suitable for BitNet adaptation.
        ssm_params = self.x_proj(x_conv)
        delta, B, C = torch.split(ssm_params, [self.d_inner, self.d_state, self.d_model], dim=-1)
        
        # Simple Gating mechanism (simulating selection)
        gate = torch.sigmoid(delta)
        y = x_conv * gate 
        
        # 4. Out Projection
        out = self.out_proj(y)
        
        # Residual connection handled in the block wrapper
        return out


class BitSpikeMambaModel(nn.Module):
    def __init__(self,
                 dim: int = 128,
                 depth: int = 2,
                 vocab_size: int = 100,
                 d_model: int = 128,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 num_layers: int = 2,
                 time_steps: int = 16,
                 neuron_config: Any = None,
                 output_head: bool = True,
                 **kwargs: Any):
        super().__init__()

        # Use d_model/num_layers as primary if provided
        self.dim = d_model if d_model is not None else dim
        self.depth = num_layers if num_layers is not None else depth
        self.use_head = output_head

        self.embedding = nn.Embedding(vocab_size, self.dim)
        
        # Stack of BitSSM Blocks
        self.layers = nn.ModuleList([
            BitSSM(d_model=self.dim, d_state=d_state, expand=expand) 
            for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.dim)
        
        self.head: nn.Module
        self.output_projection: nn.Module
        
        if self.use_head:
            self.head = BitLinear(self.dim, vocab_size)
            self.output_projection = self.head
        else:
            self.head = nn.Identity()
            self.output_projection = nn.Identity()

        self.time_steps = time_steps

    def forward(self, x: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Any:
        # x: (Batch, Seq)
        if x.dtype == torch.long or x.dtype == torch.int:
            x_emb = self.embedding(x)
        else:
            x_emb = x
            
        out = x_emb
        
        # Residual connections through SSM blocks
        for layer in self.layers:
            out = out + layer(out)

        out = self.norm(out)
        
        if self.use_head:
            logits = self.output_projection(out)
        else:
            logits = out

        if return_spikes:
            # Placeholder for spiking behavior (Phase 3 readiness)
            batch, seq, _ = logits.shape
            spikes = torch.zeros(batch, seq, logits.size(-1), device=x.device)
            mem = torch.zeros_like(spikes)
            return logits, spikes, mem

        return logits

    def print_model_size(self):
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        print(f"Model Size: {param_size / 1024**2:.3f} MB")


# Alias
BitSpikeMamba = BitSpikeMambaModel