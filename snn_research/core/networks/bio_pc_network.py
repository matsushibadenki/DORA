# ファイルパス: snn_research/core/networks/bio_pc_network.py
# 変更点:
# - get_mean_firing_rate メソッドの追加
# - forward パスでの発火率統計の記録

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, cast

from .abstract_snn_network import AbstractSNNNetwork
from ..layers.predictive_coding import PredictiveCodingLayer
from ..neurons.lif_neuron import LIFNeuron 

class BioPCNetwork(AbstractSNNNetwork):
    """
    予測符号化(PC)の原理に基づいた生物学的ニューラルネットワーク (Phase 2 仕様)。
    """

    def __init__(self,
                 layer_sizes: List[int],
                 config: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.config = config or {}
        
        net_config = self.config.get("model", {}).get("network", {})
        neuron_config = self.config.get("model", {}).get("neuron", {})
        train_config = self.config.get("training", {}).get("biologically_plausible", {})
        
        self.inference_steps = net_config.get("inference_steps", 12)
        self.sparsity = net_config.get("sparsity", 0.05)
        self.plasticity_start_step = train_config.get("plasticity_schedule", {}).get("start_step", 8)
        
        self.neuron_params = {
            "tau_mem": neuron_config.get("tau_mem", 20.0),
            "tau_adap": neuron_config.get("tau_adap", 200.0),
            "v_threshold": neuron_config.get("v_threshold", 1.0),
            "v_reset": neuron_config.get("v_reset", 0.0),
            "theta_plus": neuron_config.get("theta_plus", 0.5)
        }

        self.pc_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            layer = PredictiveCodingLayer(
                input_size=layer_sizes[i],
                hidden_size=layer_sizes[i+1],
                neuron_class=LIFNeuron,
                neuron_params=self.neuron_params,
                sparsity=self.sparsity,
                inference_steps=1,
                weight_tying=kwargs.get('weight_tying', True),
                learning=True
            )
            self.pc_layers.append(layer)
            
        self.online_learning_enabled = False
        
        # 統計用
        self.last_mean_firing_rate = 0.0

    def set_online_learning(self, enabled: bool):
        self.online_learning_enabled = enabled

    def reset_state(self) -> None:
        for m in self.modules():
            if m is self:
                continue
            reset_func = getattr(m, 'reset_state', None)
            if callable(reset_func):
                try:
                    reset_func()
                except Exception:
                    pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 1.0 
        batch_size = x.size(0)
        device = x.device

        states = [torch.zeros(batch_size, size, device=device) for size in self.layer_sizes]
        states[0] = x 

        errors: List[Optional[torch.Tensor]] = [None] * len(self.pc_layers)
        
        # 統計用カウンタ
        total_spikes = 0.0
        total_neurons = 0

        for t in range(self.inference_steps):
            new_states = [s.clone() for s in states]
            new_errors = [None] * len(self.pc_layers)
            new_states[0] = x

            for i, layer_module in enumerate(self.pc_layers):
                layer = cast(PredictiveCodingLayer, layer_module)
                
                bottom_val = states[i]
                top_val = states[i+1]
                td_error = errors[i+1] if i + 1 < len(errors) else None

                updated_top, error_bottom, _, spikes = layer(
                    bottom_up_input=bottom_val,
                    top_down_state=top_val,
                    top_down_error=td_error
                )

                new_states[i+1] = updated_top
                new_errors[i] = error_bottom
                
                # スパイク集計 (発火率計算用)
                if spikes is not None:
                    total_spikes += spikes.sum().item()
                    total_neurons += spikes.numel()
                
                # --- Online Learning ---
                if self.training and self.online_learning_enabled:
                    if t >= self.plasticity_start_step:
                        layer.update_weights(
                            bottom_input=bottom_val,
                            top_state=top_val,
                            error=error_bottom,
                            spikes=spikes
                        )

            states = new_states
            errors = new_errors # type: ignore

        # 平均発火率の更新 (全タイムステップ・全レイヤーの平均)
        if total_neurons > 0:
            self.last_mean_firing_rate = total_spikes / total_neurons
        else:
            self.last_mean_firing_rate = 0.0

        return states[-1]

    def get_mean_firing_rate(self) -> float:
        """直近のforwardパスにおける平均発火率を返す"""
        return self.last_mean_firing_rate

    def get_sparsity_loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.get_device())
        for layer in self.pc_layers:
            loss_attr = getattr(layer, 'get_sparsity_loss', 0.0)
            if callable(loss_attr):
                total_loss += loss_attr()
            else:
                total_loss += cast(torch.Tensor, torch.as_tensor(loss_attr))
        return total_loss

    def get_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")