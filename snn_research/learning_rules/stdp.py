# directory: snn_research/learning_rules
# file: stdp.py
# title: STDP (Rust-Accelerated)
# description: Rustカーネルを用いた高速STDP実装。利用不可時はPyTorch実装へフォールバック。

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

# Rust Kernel Import Strategy
try:
    import dora_kernel
    RUST_KERNEL_AVAILABLE = True
except ImportError:
    RUST_KERNEL_AVAILABLE = False

logger = logging.getLogger(__name__)

class STDP(nn.Module):
    """
    Spike-Timing-Dependent Plasticity (STDP) Learning Rule.
    Supports Rust acceleration via dora_kernel.
    """
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
               reward: Optional[float] = None) -> torch.Tensor:
        """
        重み更新を実行します。Rustカーネルが利用可能な場合は高速化されます。
        """
        # Reward modulation is applied to learning rate temporarily if present
        effective_lr = self.learning_rate
        if reward is not None:
            # Note: Rust implementation currently doesn't support complex tensor reward modulation per sample inside kernel easily.
            # For simplicity, if scalar reward is provided, we scale LR.
            # If tensor reward is needed, we fallback to PyTorch or extend Rust kernel later.
            if isinstance(reward, float) or (isinstance(reward, torch.Tensor) and reward.numel() == 1):
                effective_lr *= float(reward)
            else:
                # Complex reward map: Fallback to PyTorch logic for now to be safe
                return self._update_pytorch(weights, pre_spikes, post_spikes, pre_trace, post_trace, reward)

        if RUST_KERNEL_AVAILABLE and weights.device.type == 'cpu':
            return self._update_rust(weights, pre_spikes, post_spikes, pre_trace, post_trace, effective_lr)
        else:
            return self._update_pytorch(weights, pre_spikes, post_spikes, pre_trace, post_trace, reward)

    def _update_rust(self, weights, pre_spikes, post_spikes, pre_trace, post_trace, lr):
        """Rust Kernel Implementation"""
        # Convert to numpy (Zero-copy if possible on CPU)
        w_np = weights.detach().numpy()
        x_trace_np = pre_trace.detach().numpy()
        y_trace_np = post_trace.detach().numpy()
        x_spike_np = pre_spikes.detach().numpy()
        y_spike_np = post_spikes.detach().numpy()

        new_w_np = dora_kernel.stdp_weight_update(
            w_np, x_trace_np, y_trace_np, x_spike_np, y_spike_np,
            lr, self.A_plus, self.A_minus, self.w_min, self.w_max
        )
        
        return torch.from_numpy(new_w_np).to(weights.device)

    def _update_pytorch(self, weights, pre_spikes, post_spikes, pre_trace, post_trace, reward):
        """Original PyTorch Implementation"""
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
        
        # 3. Integrate
        delta_w = delta_w_plus - delta_w_minus
        delta_w = delta_w.mean(dim=0)
        
        if reward is not None:
            if isinstance(reward, float):
                delta_w = delta_w * reward
            else:
                # (Batch, 1, 1) broadcasting
                delta_w = delta_w * reward.view(-1, 1, 1).mean(dim=0)

        new_weights = weights + self.learning_rate * delta_w
        new_weights = torch.clamp(new_weights, self.w_min, self.w_max)
        
        return new_weights

    def update_trace(self, 
                     trace: torch.Tensor, 
                     spikes: torch.Tensor, 
                     tau: float, 
                     dt: float = 1.0) -> torch.Tensor:
        """Trace update (Currently PyTorch only - lightweight)"""
        decay = torch.exp(torch.tensor(-dt / tau, device=trace.device))
        new_trace = trace * decay + spikes
        return new_trace

# 互換性のためのエイリアス
STDPLearner = STDP
STDPRule = STDP