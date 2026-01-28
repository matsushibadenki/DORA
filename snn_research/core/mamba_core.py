# ファイルパス: snn_research/core/mamba_core.py
# Title: Spiking-MAMBA-2 (SSD & BitNet Integrated)
# Description:
#   Spiking-Mambaモデルの実装 (Mamba-2 Architecture)。
#   変更: パラレルプロジェクション、SSDロジック、Multi-Head/Group構造の採用。

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Type, cast, Optional

from .neurons import (
    AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
    TC_LIF, DualThresholdNeuron
)
from .base import BaseModel, SNNLayerNorm

# BitNetのインポート (存在チェック付き)
try:
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # BitSpikeLinearがない場合は通常のLinearを使用
    class BitSpikeLinear(nn.Linear): # type: ignore
        def __init__(self, in_features, out_features, bias=True, **kwargs):
            super().__init__(in_features, out_features, bias=bias)

from spikingjelly.activation_based import base as sj_base # type: ignore
from spikingjelly.activation_based import functional as SJ_F # type: ignore

class SpikingMamba2Block(sj_base.MemoryModule):
    """
    Spiking-MAMBA-2 Block with BitNet Weights & SSD Logic.
    Based on "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality"
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        neuron_class: Type[nn.Module], 
        neuron_params: Dict[str, Any],
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        assert self.d_inner % self.headdim == 0, "d_inner must be divisible by headdim"
        self.nheads = self.d_inner // self.headdim
        
        # Mamba-2 Projection Logic
        # in_proj produces:
        # - z (gate): d_inner
        # - x (input): d_inner
        # - B (state control): ngroups * d_state
        # - C (state control): ngroups * d_state
        # - dt (timescale): nheads
        self.d_proj_out = self.d_inner * 2 + (self.ngroups * self.d_state * 2) + self.nheads
        
        self.in_proj = BitSpikeLinear(d_model, self.d_proj_out, bias=False)
        
        # Conv1d for x, B, C (Causal Conv)
        # Groups are set to treat each channel independently (depthwise)
        # We only apply conv to x, B, C part usually, or just x. 
        # In standard Mamba-2, conv is applied to z, x, B, C before split.
        self.conv1d = nn.Conv1d(
            in_channels=self.d_proj_out - self.nheads, # dt does not undergo conv usually in some impl, but let's follow standard
            out_channels=self.d_proj_out - self.nheads,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_proj_out - self.nheads,
            padding=d_conv - 1,
        )

        # Spiking Neuron: Applied only to 'x' part
        self.lif_conv = neuron_class(features=self.d_inner, **neuron_params)

        # SSM Parameters
        # dt_bias is explicitly handled
        self.dt_bias = nn.Parameter(torch.rand(self.nheads))
        
        # A parameter (Parameter A_log for stability)
        # In Mamba-2, A is typically head-specific and scalar/diagonal
        self.A_log = nn.Parameter(torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32).repeat(1))) # Minimal init
        self.D = nn.Parameter(torch.ones(self.nheads))
        
        self.norm = SNNLayerNorm(self.d_inner) # Norm before out_proj in Mamba-2 block usually involves GroupNorm or RMSNorm
        self.out_proj = BitSpikeLinear(self.d_inner, d_model, bias=False)

    def set_stateful(self, stateful: bool) -> None:
        self.stateful = stateful
        for m in self.modules():
            if isinstance(m, sj_base.MemoryModule) and m is not self:
                m.set_stateful(stateful)

    def reset(self) -> None:
        super().reset()
        for m in self.modules():
            if isinstance(m, sj_base.MemoryModule) and m is not self:
                m.reset()

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (Batch, Length, d_model)
        """
        B, L, _ = u.shape
        
        # 1. Parallel Projection
        zxbcdt = self.in_proj(u) # (B, L, d_proj_out)
        
        # 2. Convolution (Grouped)
        # Split dt out because it usually bypasses conv or has different treatment in some variants
        # Here we assume conv applies to z, x, B, C
        conv_in = zxbcdt[:, :, :-self.nheads].transpose(1, 2) # (B, Dim, L)
        conv_out = self.conv1d(conv_in)[:, :, :L].transpose(1, 2) # Causal crop
        
        dt = zxbcdt[:, :, -self.nheads:] # (B, L, nheads)
        
        # 3. Split Parameters
        # Dimensions:
        # z: d_inner
        # x: d_inner
        # B_param: ngroups * d_state
        # C_param: ngroups * d_state
        d_inner = self.d_inner
        d_state = self.d_state
        ngroups = self.ngroups
        
        z, x, B_param, C_param = torch.split(
            conv_out, 
            [d_inner, d_inner, ngroups * d_state, ngroups * d_state], 
            dim=-1
        )
        
        # 4. Spiking Activation (Apply to x only)
        # x is the information carrier. B/C are control signals (gates).
        x_flat = x.reshape(B * L, -1)
        x_spikes = self.lif_conv(x_flat)
        x_spikes = x_spikes.reshape(B, L, d_inner)
        
        # 5. SSD (Structured State Space Duality) Logic
        # Reshape for Multi-Head
        x_reshaped = x_spikes.view(B, L, self.nheads, self.headdim)
        z_reshaped = z.view(B, L, self.nheads, self.headdim)
        
        # Broadcast B and C to heads
        # B_param: (B, L, ngroups, d_state) -> (B, L, nheads, d_state)
        ratio = self.nheads // ngroups
        B_reshaped = B_param.view(B, L, ngroups, d_state).repeat_interleave(ratio, dim=2)
        C_reshaped = C_param.view(B, L, ngroups, d_state).repeat_interleave(ratio, dim=2)
        
        # Discretization
        dt = F.softplus(dt + self.dt_bias) # (B, L, nheads)
        A = -torch.exp(self.A_log.float()) # (nheads,)
        
        # Computation of y (SSD)
        # y = SSM(A, B, C)(x)
        # In SNN context, we keep this causal and iterative to allow online processing potential,
        # though parallel scan (associative scan) is standard for Mamba.
        # Here we implement a simplified causal recurrence:
        # h_t = (1 - A*dt) * h_{t-1} + B * x_t * dt  (Simplified Euler)
        # Actually Mamba uses ZOH/Bilinear:
        #   dA = exp(A * dt)
        #   dB = (exp(A * dt) - 1)/A * B (approx) or just dt * B
        
        # Let's use the discrete recurrence form: h_t = A_bar * h_{t-1} + B_bar * x_t
        A_bar = torch.exp(A * dt) # (B, L, nheads)
        # Approximate B_bar for speed: B * dt
        B_bar = B_reshaped * dt.unsqueeze(-1) # (B, L, nheads, d_state)
        
        # Scan Loop (Conceptually SSD)
        # To strictly follow SSD matrix form, one would construct the mask matrix.
        # But for inference/SNN, explicit state update is better.
        
        h = torch.zeros(B, self.nheads, self.d_state, self.headdim, device=u.device) 
        # Note: Mamba-2 state is (B, nheads, d_state, headdim) technically for the "Dual" view (matrix mult),
        # but in recurrence it's usually (B, nheads, d_state) or (B, nheads, headdim, d_state).
        # Let's align with standard SSM: state is size d_state. x is projected to rank-1 update.
        # Actually in Mamba-2, the state is a matrix H (headdim x d_state) per head.
        
        ys = []
        for t in range(L):
            # x_t: (B, nheads, headdim)
            # B_t: (B, nheads, d_state)
            # C_t: (B, nheads, d_state)
            # A_bar_t: (B, nheads)
            
            xt_step = x_reshaped[:, t]
            Bt_step = B_bar[:, t]
            Ct_step = C_reshaped[:, t]
            At_step = A_bar[:, t].unsqueeze(-1).unsqueeze(-1) # (B, nheads, 1, 1)
            
            # State Update: H_t = A_bar * H_{t-1} + B_bar^T * x_t  (Outer product update)
            # h shape: (B, nheads, d_state, headdim)
            
            # Outer product term: (B, nheads, d_state, 1) * (B, nheads, 1, headdim)
            update = Bt_step.unsqueeze(-1) @ xt_step.unsqueeze(-2) 
            
            h = At_step * h + update
            
            # Output: y_t = h_t^T @ C_t
            # (B, nheads, headdim, d_state) @ (B, nheads, d_state, 1) -> (B, nheads, headdim, 1)
            yt_step = h.transpose(-1, -2) @ Ct_step.unsqueeze(-1)
            ys.append(yt_step.squeeze(-1))
            
        y = torch.stack(ys, dim=1) # (B, L, nheads, headdim)
        
        # D skip connection
        y = y + x_reshaped * self.D.view(1, 1, -1, 1)
        
        # 6. Gating and Output
        y = y * F.silu(z_reshaped)
        y = y.view(B, L, d_inner)
        
        y = self.norm(y)
        out = self.out_proj(y)
        
        return out

class SpikingMamba(BaseModel):
    """
    SpikingMamba (Mamba-2 Backend)
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        d_state: int, 
        d_conv: int, 
        expand: int, 
        num_layers: int, 
        time_steps: int, 
        neuron_config: Dict[str, Any], 
        headdim: int = 64,
        ngroups: int = 1,
        **kwargs: Any
    ):
        super().__init__()
        self.time_steps = time_steps

        neuron_type = neuron_config.get("type", "lif")
        neuron_params = neuron_config.copy()
        neuron_params.pop('type', None)
        
        neuron_class: Any
        
        # Neuron selection logic (Simplified for brevity)
        if neuron_type == 'lif':
            neuron_class = AdaptiveLIFNeuron
        elif neuron_type == 'izhikevich':
            neuron_class = IzhikevichNeuron
        elif neuron_type == 'glif':
            neuron_class = GLIFNeuron
            neuron_params['gate_input_features'] = d_model * expand
        elif neuron_type == 'tc_lif':
            neuron_class = TC_LIF
        elif neuron_type == 'dual_threshold':
            neuron_class = DualThresholdNeuron
        else:
             neuron_class = AdaptiveLIFNeuron

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            SpikingMamba2Block(
                d_model=d_model, 
                d_state=d_state, 
                d_conv=d_conv, 
                expand=expand, 
                neuron_class=neuron_class, 
                neuron_params=neuron_params,
                headdim=headdim,
                ngroups=ngroups
            )
            for _ in range(num_layers)
        ])
        self.norm = SNNLayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_weights()

    def _init_weights(self):
        pass

    def forward(self, input_ids: torch.Tensor, return_spikes: bool = False, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L = input_ids.shape
        device = input_ids.device
        
        SJ_F.reset_net(self)
        
        x_embed = self.embedding(input_ids)
        x = x_embed
        
        # SNN Time Step Loop
        for _ in range(self.time_steps):
            x_step = x_embed
            for layer in self.layers:
                x_step = layer(x_step)
            x = x_step
        
        x_out = self.norm(x)
        logits = self.output_projection(x_out)
        
        avg_spikes = torch.tensor(0.0, device=device)
        mem = torch.tensor(0.0, device=device) 
        
        return logits, avg_spikes, mem