# ファイルパス: snn_research/core/neurons/lif_neuron.py
# 日本語タイトル: Adaptive Leaky Integrate-and-Fire (ALIF) Neuron
# 目的・内容:
#   Objective.md Phase 2 準拠の改良版ニューロン。
#   適応的閾値（Adaptive Threshold）を導入し、発火率を自律的に抑制（<5%）する。
#   これにより「エネルギー効率」と「表現のスパース性」を向上させる。

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LIFNeuron(nn.Module):
    """
    Adaptive Leaky Integrate-and-Fire Neuron Model.
    
    Dynamics:
        tau_mem * dV/dt = -(V - V_rest) + R * I
        tau_adap * dTheta/dt = -Theta
    
    Trigger:
        If V >= V_threshold + Theta -> Spike
        Then V = V_reset, Theta = Theta + theta_plus
    """

    def __init__(
        self,
        features: int,
        tau_mem: float = 20.0,
        tau_adap: float = 200.0,  # 適応閾値の時定数（膜電位より遅い）
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        theta_plus: float = 0.5,  # 発火時に閾値をどれだけ上げるか
        dt: float = 1.0
    ):
        super().__init__()
        self.features = features
        self.tau_mem = tau_mem
        self.tau_adap = tau_adap
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.theta_plus = theta_plus
        self.dt = dt

        # State 1: Membrane Potential
        # register_buffer ensures it's part of state_dict but not a learnable parameter
        self.mem: torch.Tensor  # mypyのために型を明示
        self.register_buffer("mem", torch.zeros(1, features))
        
        # State 2: Adaptive Threshold (Theta) - これによりスパース性を強制
        self.adap_thresh: torch.Tensor  # mypyのために型を明示
        self.register_buffer("adap_thresh", torch.zeros(1, features))
        
        self.is_stateful = False

    def set_stateful(self, stateful: bool):
        self.is_stateful = stateful

    def reset_state(self):
        """状態のリセット"""
        if self.mem is not None:
            self.mem.fill_(self.v_reset)
        if self.adap_thresh is not None:
            self.adap_thresh.fill_(0.0)

    def forward(self, input_current: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_current: (Batch, Features)
        Returns:
            spikes: (Batch, Features)
            mem: (Batch, Features)
        """
        batch_size = input_current.shape[0]

        # 状態の初期化または維持
        if not self.is_stateful or self.mem.shape[0] != batch_size:
            self.mem = torch.full(
                (batch_size, self.features), self.v_reset, device=input_current.device)
            self.adap_thresh = torch.zeros(
                (batch_size, self.features), device=input_current.device)

        # --- 1. Membrane Potential Update (Euler Integration) ---
        decay_mem = self.dt / self.tau_mem
        # dV = (-(V - V_rest) + I) * (dt / tau)
        delta_v = (-(self.mem - self.v_reset) + input_current) * decay_mem
        self.mem = self.mem + delta_v

        # --- 2. Adaptive Threshold Update ---
        # 閾値成分も時間とともに減衰し、元の v_threshold に戻ろうとする
        decay_adap = self.dt / self.tau_adap
        self.adap_thresh = self.adap_thresh * (1.0 - decay_adap)

        # --- 3. Spike Generation (Effective Threshold) ---
        # 有効閾値 = ベース閾値 + 適応成分
        effective_threshold = self.v_threshold + self.adap_thresh
        
        # 勾配計算は行わない（Objective.mdの制約: Backpropなし）
        spikes = (self.mem >= effective_threshold).float()

        # --- 4. State Update after Spike ---
        # Reset Membrane Potential
        # Hard reset: V = V_reset
        self.mem = self.mem * (1.0 - spikes) + self.v_reset * spikes
        
        # Update Adaptive Threshold
        # 発火したニューロンの閾値を上げ、次の発火をしにくくする（不応期・スパース化の効果）
        self.adap_thresh = self.adap_thresh + (self.theta_plus * spikes)

        return spikes, self.mem

    @property
    def membrane_potential(self):
        return self.mem

    def reset(self):
        self.reset_state()