# snn_research/models/experimental/bit_spike_mamba.py
# Title: BitSpikeMamba v1.2 (Mypy Fix)
# Description: self.head と self.output_projection の型定義を修正し、mypyエラーを解消。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Any, Optional


def bit_quantize_weight(w: torch.Tensor, eps: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = w.abs().mean().clamp(min=eps)
    w_scaled = w / scale
    w_quant = (w_scaled).round().clamp(-1, 1)
    w_quant = (w_quant - w_scaled).detach() + w_scaled
    return w_quant, scale


class BitLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant, scale = bit_quantize_weight(self.weight)
        return F.linear(x, w_quant) * scale + (self.bias if self.bias is not None else 0)


class BitSpikeMambaModel(nn.Module):
    def __init__(self,
                 dim: int = 128,
                 depth: int = 2,
                 vocab_size: int = 100,
                 # 互換性のための追加引数
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

        # 引数の優先順位処理
        self.dim = d_model if d_model is not None else dim
        self.depth = num_layers if num_layers is not None else depth
        self.use_head = output_head

        self.embedding = nn.Embedding(vocab_size, self.dim)
        self.layers = nn.ModuleList([
            BitLinear(self.dim, self.dim) for _ in range(self.depth)
        ])

        self.norm = nn.LayerNorm(self.dim)
        
        # [Fix] 型アノテーションを追加して、BitLinear と nn.Identity の両方を許容する
        self.head: nn.Module
        self.output_projection: nn.Module
        
        if self.use_head:
            self.head = BitLinear(self.dim, vocab_size)
            self.output_projection = self.head
        else:
            self.head = nn.Identity()
            self.output_projection = nn.Identity()

        # 内部状態シミュレーション用
        self.time_steps = time_steps

    def forward(self, x: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Any:
        # x: (Batch, Seq) or (Batch, Seq, Dim)
        if x.dtype == torch.long or x.dtype == torch.int:
            x_emb = self.embedding(x)
        else:
            # Assume already embedded (e.g., coordinate inputs)
            x_emb = x
            
        out = x_emb
        for layer in self.layers:
            out = layer(out)

        # 正規化
        out = self.norm(out)
        
        # ヘッドがある場合のみ射影
        if self.use_head:
            logits = self.output_projection(out)
        else:
            logits = out # Return hidden states directly

        if return_spikes:
            # ダミーのスパイクと膜電位を返す
            batch, seq, _ = logits.shape
            spikes = torch.zeros(batch, seq, logits.size(-1), device=x.device)
            mem = torch.zeros_like(spikes)
            return logits, spikes, mem

        return logits

    def print_model_size(self):
        param_size = sum(p.nelement() * p.element_size()
                         for p in self.parameters())
        print(f"Model Size: {param_size / 1024**2:.3f} MB")


# エイリアス
BitSpikeMamba = BitSpikeMambaModel