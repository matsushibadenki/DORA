# ファイルパス: snn_research/core/neurons/lif_neuron.py
# 日本語タイトル: Adaptive Leaky Integrate-and-Fire (ALIF) Neuron (Full Fix)
# 目的・内容:
#   プロパティ追加とサロゲート勾配の定義。
#   Mypy型エラー修正のため型ヒントを追加。

import torch
import torch.nn as nn
from typing import Tuple, Optional, Any, Union
from torch import Tensor


class SurrogateHeaviside(torch.autograd.Function):
    """
    微分不可能なHeaviside関数のためのサロゲート勾配定義。
    Backpropagation時にATan関数の勾配で近似する。
    """

    @staticmethod
    def forward(ctx: Any, input_tensor: Tensor, alpha: float = 2.0) -> Tensor:
        ctx.save_for_backward(input_tensor)
        ctx.alpha = alpha
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, None]:
        (input_tensor,) = ctx.saved_tensors
        alpha = ctx.alpha
        # ATanの導関数による勾配近似
        grad_input = grad_output * (1 / (1 + (alpha * input_tensor).pow(2)))
        return grad_input, None


class LIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire Neuron Model.
    """

    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        tau_adap: float = 200.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        theta_plus: float = 0.5,
        dt: float = 1.0,
        trainable_tau: bool = False # 互換性引数
    ):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.tau_adap = tau_adap
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.theta_plus = theta_plus
        self.dt = dt

        # State tensors
        # 型ヒントを明示してMypyエラーを解消
        self.mem: Tensor
        self.adap_thresh: Tensor

        self.register_buffer("mem", torch.zeros(1, features))
        self.register_buffer("adap_thresh", torch.zeros(1, features))
        
        self.is_stateful = False
        self.spike_fn = SurrogateHeaviside.apply

    def set_stateful(self, stateful: bool):
        self.is_stateful = stateful

    def reset_state(self):
        """状態のリセット"""
        if self.mem is not None:
            self.mem.fill_(self.v_reset)
        if self.adap_thresh is not None:
            self.adap_thresh.fill_(0.0)

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = input_current.shape[0]

        # 状態の初期化または維持
        # self.memの型ヒントがあるためMypyエラーは解消される
        if not self.is_stateful or self.mem.shape[0] != batch_size:
            self.mem = torch.full(
                (batch_size, self.features), self.v_reset, device=input_current.device)
            self.adap_thresh = torch.zeros(
                (batch_size, self.features), device=input_current.device)

        # Dynamics
        decay_mem = self.dt / self.tau_mem
        delta_v = (-(self.mem - self.v_reset) + input_current) * decay_mem
        self.mem = self.mem + delta_v

        decay_adap = self.dt / self.tau_adap
        self.adap_thresh = self.adap_thresh * (1.0 - decay_adap)

        effective_threshold = self.v_threshold + self.adap_thresh
        
        # Spike generation with Surrogate Gradient
        # ここで spike_fn を通すことで、学習時に勾配が伝播するようになる
        spikes = self.spike_fn(self.mem - effective_threshold)

        # State Update
        self.mem = self.mem * (1.0 - spikes) + self.v_reset * spikes
        self.adap_thresh = self.adap_thresh + (self.theta_plus * spikes)

        return spikes, self.mem

    @property
    def membrane_potential(self) -> torch.Tensor:
        """後方互換性のためのプロパティ"""
        return self.mem

    def reset(self):
        self.reset_state()