# directory: snn_research/learning_rules
# file: stdp.py
# title: R-STDP (Reward-Modulated STDP)
# description: 報酬信号を受け取り、Rustカーネルに渡すように更新。互換性エイリアスを追加。

import torch
import torch.nn as nn
from typing import Optional
import logging

try:
    import dora_kernel
    RUST_KERNEL_AVAILABLE = True
except ImportError:
    RUST_KERNEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class STDP(nn.Module):
    def __init__(self, 
                 learning_rate: float = 1e-4,
                 w_min: float = 0.0,
                 w_max: float = 1.0,
                 tau_pre: float = 20.0,
                 tau_post: float = 20.0,
                 modulatory_factor: float = 1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.w_min = w_min
        self.w_max = w_max
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.A_plus = 1.0 * modulatory_factor
        self.A_minus = 1.05 * modulatory_factor

    def update(self, 
               weights: torch.Tensor, 
               pre_spikes: torch.Tensor, 
               post_spikes: torch.Tensor, 
               pre_trace: torch.Tensor, 
               post_trace: torch.Tensor,
               reward: float = 1.0) -> torch.Tensor:
        """
        R-STDP update.
        Args:
            reward: 報酬信号 (Scalar). 
                    1.0 = 通常のUnsupervised STDP (LTP/LTD)
                    >1.0 = 強化 (Strong reinforcement)
                    <0.0 = 罰 (Anti-STDP / Punishment)
                    0.0 = 学習なし
        """
        if RUST_KERNEL_AVAILABLE and weights.device.type == 'cpu':
            return self._update_rust(weights, pre_spikes, post_spikes, pre_trace, post_trace, self.learning_rate, reward)
        else:
            return self._update_pytorch(weights, pre_spikes, post_spikes, pre_trace, post_trace, reward)

    def _update_rust(self, weights, pre_spikes, post_spikes, pre_trace, post_trace, lr, reward):
        # Zero-copy conversion
        w_np = weights.detach().numpy()
        x_trace_np = pre_trace.detach().numpy()
        y_trace_np = post_trace.detach().numpy()
        x_spike_np = pre_spikes.detach().numpy()
        y_spike_np = post_spikes.detach().numpy()

        # Call Rust Kernel with Reward
        # 注意: Rust側のシグネチャに合わせて引数を渡す
        new_w_np = dora_kernel.stdp_weight_update(
            w_np, x_trace_np, y_trace_np, x_spike_np, y_spike_np,
            lr, self.A_plus, self.A_minus, self.w_min, self.w_max,
            float(reward) # Pass reward as float
        )
        
        return torch.from_numpy(new_w_np).to(weights.device)

    def _update_pytorch(self, weights, pre_spikes, post_spikes, pre_trace, post_trace, reward):
        # 1. LTP
        delta_w_plus = self.A_plus * torch.matmul(
            post_spikes.unsqueeze(2), 
            pre_trace.unsqueeze(1)
        )
        # 2. LTD
        delta_w_minus = self.A_minus * torch.matmul(
            post_trace.unsqueeze(2), 
            pre_spikes.unsqueeze(1)
        )
        # 3. Integrate & Modulate
        delta_w = delta_w_plus - delta_w_minus
        delta_w = delta_w.mean(dim=0)
        
        # Reward Modulation
        delta_w = delta_w * reward

        new_weights = weights + self.learning_rate * delta_w
        new_weights = torch.clamp(new_weights, self.w_min, self.w_max)
        
        return new_weights

    def update_trace(self, trace, spikes, tau, dt=1.0):
        decay = torch.exp(torch.tensor(-dt / tau, device=trace.device))
        new_trace = trace * decay + spikes
        return new_trace

# --- 修正: 互換性エイリアスを追加 ---
STDPLearner = STDP
STDPRule = STDP