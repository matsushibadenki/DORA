# ファイルパス: snn_research/models/bio/simple_network.py
# Title: BioSNN with Learning Reset (Fixed Cast)
# Description: mypyエラー "Tensor not callable" を解消するためのキャスト追加。

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, cast

import logging
from snn_research.core.base import BaseModel
from snn_research.learning_rules.base_rule import BioLearningRule

logger = logging.getLogger(__name__)


class BioSNN(BaseModel):
    """
    生物学的妥当性を備えた多層SNN。
    """

    def __init__(
        self,
        layer_sizes: List[int],
        neuron_params: Dict[str, Any],
        synaptic_rule: BioLearningRule,
        homeostatic_rule: Optional[BioLearningRule] = None,
        sparsification_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.neuron_params = neuron_params

        self.tau_mem = float(neuron_params.get('tau_mem', 20.0))
        self.v_threshold = float(neuron_params.get('v_threshold', 1.0))
        self.v_reset = float(neuron_params.get('v_reset', 0.0))
        self.dt = float(neuron_params.get('dt', 1.0))
        self.noise_std = float(neuron_params.get('noise_std', 0.1))

        self.weights = nn.ParameterList()
        # 型ヒントを nn.ModuleList に明示
        self.synaptic_rules: nn.ModuleList = nn.ModuleList()
        self.mem_potentials: List[torch.Tensor] = []

        import copy
        for i in range(len(layer_sizes) - 1):
            gain = 2.0
            w_init = torch.randn(
                layer_sizes[i], layer_sizes[i+1]) * (gain / (layer_sizes[i] ** 0.5))
            self.weights.append(nn.Parameter(w_init))
            self.synaptic_rules.append(copy.deepcopy(synaptic_rule))

        config = sparsification_config or {}
        self.sparsification_enabled = config.get("enabled", False)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.mem_potentials = []
        for size in self.layer_sizes[1:]:
            self.mem_potentials.append(
                torch.zeros(batch_size, size, device=device))

    def reset_learning_rules(self) -> None:
        """全てのシナプス学習則の内部状態をリセットする"""
        for rule in self.synaptic_rules:
            # mypyが rule を Tensor と誤認する場合の対策: Anyにキャスト
            rule_obj = cast(Any, rule)
            if hasattr(rule_obj, 'reset_state'):
                rule_obj.reset_state()
            elif hasattr(rule_obj, 'reset'):
                rule_obj.reset()

    def update_weights(
        self, 
        all_layer_spikes: List[torch.Tensor], 
        optional_params: Optional[Dict[str, Any]] = None
    ) -> None:
        
        params = optional_params or {}

        for i in range(len(self.weights)):
            pre_spikes = all_layer_spikes[i]
            post_spikes = all_layer_spikes[i+1]
            
            # ModuleListからの取得時もキャスト推奨
            rule: BioLearningRule = cast(BioLearningRule, self.synaptic_rules[i])

            dw, _ = rule.update(
                pre_spikes, 
                post_spikes,
                self.weights[i], 
                **params
            )

            if dw is not None:
                with torch.no_grad():
                    self.weights[i].add_(dw)
                    self.weights[i].clamp_(-5.0, 5.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        batch_size = x.shape[0]
        device = x.device

        if not self.mem_potentials or self.mem_potentials[0].shape[0] != batch_size:
            self.reset_state(batch_size, device)

        spikes_history = [x]
        current_input = x

        for i, weight in enumerate(self.weights):
            current = torch.matmul(current_input, weight)

            if self.training and self.noise_std > 0:
                noise = torch.randn_like(current) * self.noise_std
                current = current + noise

            decay = 1.0 - (self.dt / self.tau_mem)
            self.mem_potentials[i] = self.mem_potentials[i] * decay + current

            spikes = (self.mem_potentials[i] >= self.v_threshold).float()

            self.mem_potentials[i] = self.mem_potentials[i] - \
                (spikes * self.v_threshold)

            current_input = spikes
            spikes_history.append(spikes)

        return current_input, spikes_history