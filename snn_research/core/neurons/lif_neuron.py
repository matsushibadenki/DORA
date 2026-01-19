# ファイルパス: snn_research/core/neurons/lif_neuron.py
# 日本語タイトル: Leaky Integrate-and-Fire (LIF) Neuron
# 目的・内容:
#   標準的なスパイキングニューロンモデル。
#   膜電位の積分、リーク、発火、不応期をシミュレートする。
#   PyTorchのautogradに対応しつつ、物理的なダイナミクスを保持。

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire Neuron Model.

    Dynamics:
        tau * dV/dt = -(V - V_rest) + R * I
        If V >= V_threshold -> Spike, V = V_reset
    """

    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        dt: float = 1.0
    ):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.dt = dt

        # State (mem: Membrane Potential)
        # register_buffer ensures it's part of state_dict but not a learnable parameter
        self.mem: torch.Tensor
        self.register_buffer("mem", torch.zeros(1, features))
        self.is_stateful = False

    def set_stateful(self, stateful: bool):
        """
        Trueの場合、forward間で膜電位を保持する（RNN的挙動）。
        Falseの場合、毎回リセットする（Feedforward的挙動）。
        Kernelでは通常Trueで使用される。
        """
        self.is_stateful = stateful

    def reset_state(self):
        """膜電位のリセット"""
        if self.mem is not None:
            self.mem.fill_(self.v_reset)

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_current (Tensor): Synaptic input current (Batch, Features)
        Returns:
            spikes (Tensor): Binary spikes (1.0 or 0.0)
            mem (Tensor): Membrane potential
        """
        batch_size = input_current.shape[0]

        # 状態の初期化または維持
        if not self.is_stateful or self.mem.shape[0] != batch_size:
            self.mem = torch.full(
                (batch_size, self.features), self.v_reset, device=input_current.device)

        # Euler Integration
        # dV = (-(V - V_rest) + I) * (dt / tau)
        decay = self.dt / self.tau_mem
        delta_v = (-(self.mem - self.v_reset) + input_current) * decay

        # Update membrane potential
        self.mem = self.mem + delta_v

        # Spike generation (Heaviside step function)
        # 勾配計算のためにSurrogate Gradientを使うのが一般的だが、
        # ここではResearch OSとしての「現象シミュレーション」を優先し、単純な閾値判定を行う。
        # (Forward-ForwardやSTDPでは勾配を使わないため問題ない)
        spikes = (self.mem >= self.v_threshold).float()

        # Reset mechanism (Soft reset or Hard reset)
        # Hard reset: V = V_reset
        self.mem = self.mem * (1.0 - spikes) + self.v_reset * spikes

        return spikes, self.mem

    @property
    def membrane_potential(self):
        return self.mem

    @property
    def v(self):
        return self.mem

    def reset(self):
        self.reset_state()
