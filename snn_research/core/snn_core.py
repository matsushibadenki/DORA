# snn_research/core/snn_core.py
# Title: Spiking Neural Substrate (Core)
# Description: Mypy型安全性を強化し、未実装メソッドを追加した修正版

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast, Union

import torch
import torch.nn as nn
from torch import Tensor

from snn_research.core.neurons.lif_neuron import LIFNeuron
from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)


class SynapticProjection(nn.Module):
    """ニューロン集団間のシナプス結合と可塑性管理"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        plasticity_rule: Optional[PlasticityRule] = None
    ) -> None:
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.plasticity_rule = plasticity_rule
        self.plasticity_state: Dict[str, Any] = {}
        
        # 初期化
        nn.init.orthogonal_(self.synapse.weight, gain=1.4)

    def forward(self, x: Tensor) -> Tensor:
        return self.synapse(x)

    def apply_plasticity(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        if self.plasticity_rule is None:
            return {}

        logs: Dict[str, Any] = {}
        with torch.no_grad():
            delta_w, logs = self.plasticity_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                current_weights=self.synapse.weight.data,
                local_state=self.plasticity_state,
                **kwargs
            )

            if delta_w is not None:
                self.synapse.weight.data += delta_w
                self.synapse.weight.data.clamp_(-5.0, 5.0)

        return logs
    
    def reset_state(self) -> None:
        self.plasticity_state.clear()


class SpikingNeuralSubstrate(nn.Module):
    """
    Neuromorphic OS Kernel.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.time_step: int = 0
        self.dt: float = config.get("dt", 1.0)

        self.neuron_groups: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.topology: List[Dict[str, str]] = []
        self.prev_spikes: Dict[str, Optional[Tensor]] = {}
        
        # 統計用
        self.total_spike_count: int = 0

        logger.info("⚡ SpikingNeuralSubstrate initialized.")

    def add_neuron_group(
        self,
        name: str,
        num_neurons: int,
        neuron_model: Optional[nn.Module] = None
    ) -> None:
        if neuron_model is None:
            neuron_model = LIFNeuron(
                features=num_neurons,
                tau_mem=self.config.get("tau_mem", 20.0),
                v_threshold=self.config.get("threshold", 1.0),
                dt=self.dt
            )

        if hasattr(neuron_model, "set_stateful"):
            cast(Any, neuron_model).set_stateful(True)

        self.neuron_groups[name] = neuron_model.to(self.device)
        self.prev_spikes[name] = None

    def add_projection(
        self,
        name: str,
        source: str,
        target: str,
        plasticity_rule: Optional[PlasticityRule] = None
    ) -> None:
        if source not in self.neuron_groups or target not in self.neuron_groups:
            raise ValueError(f"Source {source} or Target {target} not found.")

        src_module = cast(Any, self.neuron_groups[source])
        tgt_module = cast(Any, self.neuron_groups[target])

        # feature属性がない場合に対応するためのフォールバック
        src_dim = getattr(src_module, 'features', getattr(src_module, 'out_features', None))
        tgt_dim = getattr(tgt_module, 'features', getattr(tgt_module, 'out_features', None))
        
        if src_dim is None or tgt_dim is None:
             raise ValueError(f"Could not determine dimensions for {source} -> {target}")

        projection = SynapticProjection(int(src_dim), int(tgt_dim), plasticity_rule)
        self.projections[name] = projection.to(self.device)
        self.topology.append({"name": name, "src": source, "tgt": target})

    def forward(self, x: Union[Tensor, Dict[str, Tensor]], **kwargs: Any) -> Tensor:
        """
        nn.Moduleの標準呼び出し。
        入力がTensorの場合は最初のニューロン層への入力とみなす。
        """
        inputs: Dict[str, Tensor] = {}
        if isinstance(x, dict):
            inputs = x
        elif torch.is_tensor(x):
            # 入力層を自動推定 (Retina, Inputなど) または登録順の最初
            input_names = [name for name in self.neuron_groups.keys() if "retina" in name.lower() or "input" in name.lower()]
            target_layer = input_names[0] if input_names else list(self.neuron_groups.keys())[0]
            inputs[target_layer] = x
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")

        results = self.forward_step(inputs, **kwargs)
        
        # エージェントが期待する出力形式 (Tensor) に合わせる
        spikes = results["spikes"]
        output_names = [name for name in spikes.keys() if "output" in name.lower() or "motor" in name.lower() or "readout" in name.lower()]
        target_output = output_names[0] if output_names else list(spikes.keys())[-1]
        
        return spikes[target_output]

    def forward_step(
        self,
        external_inputs: Dict[str, Tensor],
        **kwargs: Any
    ) -> Dict[str, Any]:
        self.time_step += 1
        
        batch_size = 1
        for inp in external_inputs.values():
            batch_size = inp.shape[0]
            break

        self._initialize_prev_spikes_if_needed(batch_size)
        current_inputs = self._integrate_inputs(external_inputs, batch_size)
        current_spikes = self._update_neuron_dynamics(current_inputs, batch_size)
        self._apply_plasticity(current_spikes, **kwargs)
        self.prev_spikes = cast(Dict[str, Optional[Tensor]], current_spikes)

        # 統計更新
        for s in current_spikes.values():
            self.total_spike_count += int(s.sum().item())

        return {"spikes": current_spikes}

    def _initialize_prev_spikes_if_needed(self, batch_size: int) -> None:
        for name, group in self.neuron_groups.items():
            prev = self.prev_spikes.get(name)
            # mypyのための型絞り込み：prevがNoneでない場合のみshapeチェックを行う
            if prev is None or prev.shape[0] != batch_size:
                group_module = cast(Any, group)
                num_neurons = int(getattr(group_module, 'features', getattr(group_module, 'out_features', 0)))
                if num_neurons > 0:
                    self.prev_spikes[name] = torch.zeros(
                        batch_size, num_neurons, device=self.device
                    )

    def _integrate_inputs(self, external_inputs: Dict[str, Tensor], batch_size: int) -> Dict[str, Tensor]:
        current_inputs: Dict[str, Tensor] = {}

        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module = self.projections[proj_name]
            src_spikes_prev = self.prev_spikes.get(src_name)
            
            if src_spikes_prev is None:
                 continue 

            synaptic_current = proj_module(src_spikes_prev)

            if tgt_name not in current_inputs:
                current_inputs[tgt_name] = synaptic_current
            else:
                current_inputs[tgt_name] = current_inputs[tgt_name] + synaptic_current

        for group_name, inp in external_inputs.items():
            if group_name in self.neuron_groups:
                inp = inp.to(self.device)
                if group_name not in current_inputs:
                    current_inputs[group_name] = inp
                else:
                    if current_inputs[group_name].shape == inp.shape:
                        current_inputs[group_name] = current_inputs[group_name] + inp
        
        return current_inputs

    def _update_neuron_dynamics(self, current_inputs: Dict[str, Tensor], batch_size: int) -> Dict[str, Tensor]:
        current_spikes: Dict[str, Tensor] = {}

        for name, group in self.neuron_groups.items():
            if name in current_inputs:
                input_current = current_inputs[name]
            else:
                group_module = cast(Any, group)
                num_neurons = int(getattr(group_module, 'features', getattr(group_module, 'out_features', 0)))
                input_current = torch.zeros(
                    batch_size, num_neurons, device=self.device
                )

            spikes, _ = group(input_current)
            current_spikes[name] = spikes

        return current_spikes

    def _apply_plasticity(self, current_spikes: Dict[str, Tensor], **kwargs: Any) -> None:
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module = cast(SynapticProjection, self.projections[proj_name])

            if getattr(proj_module, 'plasticity_rule', None) is not None:
                src_spikes_prev = self.prev_spikes.get(src_name)
                tgt_spikes_curr = current_spikes.get(tgt_name)
                
                if src_spikes_prev is not None and tgt_spikes_curr is not None:
                    proj_module.apply_plasticity(
                        pre_spikes=src_spikes_prev,
                        post_spikes=tgt_spikes_curr,
                        dt=self.dt,
                        **kwargs
                    )

    def reset_state(self) -> None:
        self.time_step = 0
        self.total_spike_count = 0
        self.prev_spikes = {}
        for name, group in self.neuron_groups.items():
            if hasattr(group, 'reset'):
                cast(Any, group).reset()
            self.prev_spikes[name] = None
        
        for proj in self.projections.values():
            if hasattr(proj, 'reset_state'):
                cast(Any, proj).reset_state()

        logger.info("🔄 Substrate state reset.")
    
    # --- Added missing methods referenced by Agent/Report scripts ---
    def get_firing_rates(self) -> Dict[str, float]:
        rates = {}
        for name, spikes in self.prev_spikes.items():
            if spikes is not None:
                rates[name] = float(spikes.float().mean().item())
            else:
                rates[name] = 0.0
        return rates

    def get_total_spikes(self) -> int:
        return self.total_spike_count

# Backward Compatibility Alias
SNNCore = SpikingNeuralSubstrate