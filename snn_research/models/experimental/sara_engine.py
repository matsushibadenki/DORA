# directory: snn_research/models/experimental
# file: sara_engine.py
# purpose: SARA Engine v7.0 [Rust Integration: Accelerated SNN & Mamba]

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, List, Optional

# Rustカーネルのインポート
try:
    import dora_kernel
    RUST_AVAILABLE = True
    print("[SARA] Rust kernel loaded successfully. 🚀")
except ImportError:
    dora_kernel = None
    RUST_AVAILABLE = False
    print("[SARA] Warning: dora_kernel not found. Using slow Python fallback. 🐢")

# --- 1. Stable Surrogate Gradient (不変) ---
class FastSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, scale=25.0):
        ctx.scale = scale
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output / (ctx.scale * input.abs() + 1.0) ** 2
        return grad_input, None

def surrogate_spike(input):
    return FastSigmoid.apply(input)

# --- 2. SNN Encoder (Rust Accelerated) ---
class SNNEncoder(nn.Module):
    def __init__(self, input_dim: int, n_neurons: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, n_neurons)
        self.ln = nn.LayerNorm(n_neurons)
        self.n_neurons = n_neurons
        self.tau = 2.0
        self.v_th = 1.0
        
    def forward(self, x: torch.Tensor, time_steps: int = 20) -> Tuple[torch.Tensor, float]:
        batch_size = x.size(0)
        current = self.fc(x)
        current = self.ln(current) # (Batch, Neurons)
        
        spikes_list = []
        
        # Rustカーネルが利用可能 かつ CPU実行時のみ使用 (GPU時はPyTorchの方が速い場合があるため)
        if RUST_AVAILABLE and x.device.type == 'cpu':
            # Rust用にNumpyへ変換 (オーバーヘッドはあるが計算量が多い場合は有利)
            curr_np = current.detach().numpy().astype(np.float32)
            v_np = np.zeros_like(curr_np)
            
            for _ in range(time_steps):
                # Rustで一括更新 (In-place update for v_np)
                spikes_np = dora_kernel.update_lif_neurons(
                    v_np, curr_np, self.tau, self.v_th, 1.0
                )
                # Tensorに戻す
                spikes_list.append(torch.from_numpy(spikes_np))
        else:
            # Python Fallback
            v = torch.zeros_like(current)
            for _ in range(time_steps):
                v = v + (current - v) / self.tau
                spike = surrogate_spike(v - self.v_th)
                spikes_list.append(spike)
                v = v - (self.v_th * spike)
            
        spikes_stack = torch.stack(spikes_list, dim=1) # (B, T, N)
        firing_rate = spikes_stack.mean().item()
        return spikes_stack, firing_rate

# --- 3. Spiking Mamba Memory Block (Rust Hybrid) ---
class SpikingMambaBlock(nn.Module):
    # ... (__init__ は変更なし) ...
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + self.d_state * 2 + self.dt_rank)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.act = nn.SiLU()

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.norm = nn.LayerNorm(self.d_inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        projected = self.in_proj(x)
        d_inner = self.d_inner
        z, x_in, B_ssm, C_ssm, dt_rank = torch.split(
            projected, 
            [d_inner, d_inner, self.d_state, self.d_state, self.dt_rank], 
            dim=-1
        )
        
        x_conv = x_in.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :L]
        x_conv = self.act(x_conv.transpose(1, 2)) # (B, L, D)

        dt = F.softplus(self.dt_proj(dt_rank))
        A = -torch.exp(self.A_log) # (D, N)
        
        # --- Rust Kernel Call (Batch Accelerated) ---
        if RUST_AVAILABLE and x.device.type == 'cpu':
            # Contiguousなメモリレイアウトを確保するために .contiguous() を呼ぶ
            u_np = x_conv.detach().numpy() # (B, L, D)
            dt_np = dt.detach().numpy()    # (B, L, D)
            A_np = A.detach().numpy()      # (D, N)
            B_np = B_ssm.detach().numpy()  # (B, L, N)
            C_np = C_ssm.detach().numpy()  # (B, L, N)
            
            # バッチごとではなく、全体を一括で渡す (Parallelism happens in Rust)
            y_np = dora_kernel.fast_selective_scan(
                u_np, dt_np, A_np, B_np, C_np
            )
            y = torch.from_numpy(y_np)
            
        else:
            # Python Fallback
            h = torch.zeros(B, self.d_inner, self.d_state, device=x.device)
            ys = []
            for t in range(L):
                dt_t = dt[:, t, :]
                B_t = B_ssm[:, t, :]
                C_t = C_ssm[:, t, :]
                x_t = x_conv[:, t, :]
                
                dA = torch.exp(A.unsqueeze(0) * dt_t.unsqueeze(-1))
                dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)
                
                h = dA * h + dB * x_t.unsqueeze(-1)
                y_t = torch.sum(h * C_t.unsqueeze(1), dim=-1)
                ys.append(y_t)
            y = torch.stack(ys, dim=1)
        
        y = y + x_conv * self.D
        y = y * F.silu(z)
        
        final_memory = y[:, -1, :]
        out = self.out_proj(self.norm(final_memory))
        
        return out

# --- 4. Recursive Reasoning (不変) ---
class RecursiveMeaningLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        
    def forward(self, m: torch.Tensor, max_depth: int = 5) -> Tuple[torch.Tensor, int]:
        h = torch.zeros(m.size(0), self.cell.hidden_size, device=m.device)
        h = self.cell(m, h)
        for i in range(max_depth - 1):
            h = self.cell(m, h) 
        return self.ln(h), max_depth

# --- 5. SARA Engine v7.0 (Integrated) ---
class SARAEngine(nn.Module):
    def __init__(self, input_dim: int = 784, n_encode_neurons: int = 128, 
                 d_legendre: int = 64, d_meaning: int = 128, n_output: int = 10):
        super().__init__()
        self.encoder = SNNEncoder(input_dim, n_encode_neurons)
        self.memory = SpikingMambaBlock(d_model=n_encode_neurons, d_state=16, d_conv=4, expand=2)
        self.rlm = RecursiveMeaningLayer(n_encode_neurons, d_meaning)
        self.decoder = nn.Linear(d_meaning, n_output)
        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        spikes, _ = self.encoder(x)
        m = self.memory(spikes)
        z, _ = self.rlm(m, max_depth=5)
        return z

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
        spikes, rate = self.encoder(x)
        m = self.memory(spikes)
        z, depth = self.rlm(m, max_depth=5)
        logits = self.decoder(z)
        return logits, rate, 0.0

    def compute_ff_loss(self, x_pos: torch.Tensor, x_neg: torch.Tensor, threshold: float = 2.0) -> torch.Tensor:
        z_pos = self.forward_features(x_pos)
        z_neg = self.forward_features(x_neg)
        g_pos = z_pos.pow(2).mean(dim=1)
        g_neg = z_neg.pow(2).mean(dim=1)
        loss_pos = F.softplus(threshold - g_pos).mean()
        loss_neg = F.softplus(g_neg - threshold).mean()
        return loss_pos + loss_neg