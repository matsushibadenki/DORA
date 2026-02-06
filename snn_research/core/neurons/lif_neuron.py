# snn_research/core/neurons/lif_neuron.py
# Title: LIF Neuron (Type Annotated)
# Description: クラスレベルの型アノテーションを追加し、mypyエラーを解消。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Union
from torch import Tensor

class SurrogateHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input_tensor: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        alpha = ctx.alpha
        grad_input = grad_output * (1 / (1 + (alpha * input_tensor).pow(2)))
        return grad_input, None

class LIFNeuron(nn.Module):
    # [Fix] Explicit class-level annotations for buffers
    mem: Tensor
    adap_thresh: Tensor

    def __init__(
        self,
        features: int,
        tau_mem: float = 100.0,
        tau_adap: float = 200.0,
        v_threshold: float = 0.5,
        v_reset: float = 0.0,
        theta_plus: float = 0.5,
        dt: float = 1.0,
        trainable_tau: bool = False
    ):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.tau_adap = tau_adap
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.theta_plus = theta_plus
        self.dt = dt

        self.register_buffer("mem", torch.zeros(1, features))
        self.register_buffer("adap_thresh", torch.zeros(1, features))
        
        self.is_stateful = False
        self.spike_fn = SurrogateHeaviside.apply

    def set_stateful(self, stateful: bool):
        self.is_stateful = stateful

    def reset_state(self):
        if self.mem is not None:
            self.mem.fill_(self.v_reset)
        if self.adap_thresh is not None:
            self.adap_thresh.fill_(0.0)

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_current.shape[0]

        # [Fix] Type is now inferred correctly from class annotation
        if not self.is_stateful or self.mem.shape[0] != batch_size:
            self.mem = torch.full(
                (batch_size, self.features), self.v_reset, device=input_current.device)
            self.adap_thresh = torch.zeros(
                (batch_size, self.features), device=input_current.device)

        decay_mem = self.dt / self.tau_mem
        decay_adap = self.dt / self.tau_adap

        factor_mem = 1.0 - decay_mem
        
        # Use explicit variable for readability if needed, but self.mem is typed
        mem = self.mem
        
        mem.mul_(factor_mem)
        mem.add_(input_current, alpha=decay_mem)
        
        if self.v_reset != 0.0:
            mem.add_(self.v_reset * decay_mem)

        self.adap_thresh.mul_(1.0 - decay_adap)

        effective_threshold = self.adap_thresh + self.v_threshold
        mem_shift = mem - effective_threshold
        
        if torch.is_grad_enabled():
            spikes = self.spike_fn(mem_shift)
        else:
            spikes = (mem_shift > 0).float()

        if spikes.any():
            mask = spikes.bool()
            mem.masked_fill_(mask, self.v_reset)
            self.adap_thresh.add_(spikes, alpha=self.theta_plus)

        return spikes, mem

    @property
    def membrane_potential(self) -> torch.Tensor:
        return self.mem

    def reset(self):
        self.reset_state()