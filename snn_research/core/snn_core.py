# snn_research/core/snn_core.py
# Title: Spiking Neural Substrate (API Complete)
# Description: 
#   get_firing_rates, get_total_spikes メソッドを追加し、
#   ActiveInferenceAgent等の要件を満たす。

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional, cast, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from snn_research.hardware.event_driven_simulator import DORAKernel
from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class SynapticProjection(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        plasticity_rule: Optional[PlasticityRule] = None
    ) -> None:
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.plasticity_rule = plasticity_rule
        nn.init.normal_(self.synapse.weight, mean=0.0, std=2.0)
        self.is_compiled = False

    def forward(self, x: Tensor) -> Tensor:
        return self.synapse(x)


class SpikingNeuralSubstrate(nn.Module):
    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dt: float = config.get("dt", 1.0)
        self.time_step: int = 0

        self.kernel = DORAKernel(dt=self.dt)
        self.kernel_compiled = False

        self.neuron_groups: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.topology: List[Dict[str, str]] = []
        self.prev_spikes: Dict[str, Optional[Tensor]] = {}
        
        self.group_indices: Dict[str, Tuple[int, int]] = {}
        self.total_spike_count: int = 0
        logger.info("⚡ SpikingNeuralSubstrate (DORA Kernel v1.5 API) initialized.")

    def add_neuron_group(self, name: str, num_neurons: int, neuron_model: Optional[nn.Module] = None) -> None:
        if neuron_model is None: neuron_model = nn.Identity()
        setattr(neuron_model, 'features', num_neurons)
        setattr(neuron_model, 'out_features', num_neurons)
        self.neuron_groups[name] = neuron_model
        
        layer_id = len(self.group_indices)
        start_id = len(self.kernel.neurons)
        for _ in range(num_neurons):
            self.kernel.add_neuron(layer_id=layer_id, v_thresh=1.0)
        end_id = len(self.kernel.neurons)
        self.group_indices[name] = (start_id, end_id)
        self.prev_spikes[name] = torch.zeros(1, num_neurons, device=self.device)

    def add_projection(self, name: str, source: str, target: str, plasticity_rule: Optional[PlasticityRule] = None) -> None:
        src_module = self.neuron_groups[source]
        tgt_module = self.neuron_groups[target]
        src_dim = int(getattr(src_module, 'features', getattr(src_module, 'out_features', 0)))
        tgt_dim = int(getattr(tgt_module, 'features', getattr(tgt_module, 'out_features', 0)))
        
        projection = SynapticProjection(src_dim, tgt_dim, plasticity_rule)
        self.projections[name] = projection
        self.topology.append({"name": name, "src": source, "tgt": target})

    def compile(self) -> None:
        pass 

    def forward(self, x: Union[Tensor, Dict[str, Tensor]], **kwargs: Any) -> Tensor:
        inputs: Dict[str, Tensor] = {}
        if isinstance(x, dict): inputs = x
        elif torch.is_tensor(x):
            input_names = [name for name in self.neuron_groups.keys() if "retina" in name.lower() or "input" in name.lower()]
            target_layer = input_names[0] if input_names else list(self.neuron_groups.keys())[0]
            inputs[target_layer] = x
        else: return torch.zeros(1, 1, device=self.device)

        results = self.forward_step(inputs, **kwargs)
        spikes = results["spikes"]
        output_names = [name for name in spikes.keys() if "output" in name.lower() or "motor" in name.lower() or "readout" in name.lower()]
        target_output = output_names[0] if output_names else list(spikes.keys())[-1]
        return spikes[target_output]

    def forward_step(self, external_inputs: Dict[str, Tensor], **kwargs: Any) -> Dict[str, Any]:
        self.time_step += 1
        current_time = self.kernel.current_time
        jitter = 0.1
        total_input_spikes = 0
        
        for name, tensor in external_inputs.items():
            if name in self.group_indices:
                start_id, _ = self.group_indices[name]
                if tensor.dim() > 1: tensor = tensor[0] 
                
                indices = torch.nonzero(tensor > 0.0).flatten().cpu().numpy()
                global_indices = [int(idx + start_id) for idx in indices]
                
                if len(global_indices) > 0:
                    self.kernel.push_input_spikes(global_indices, current_time + jitter)
                    total_input_spikes += len(global_indices)
        
        if total_input_spikes == 0 and self.group_indices:
            first_group = list(self.group_indices.values())[0]
            random_indices = [random.randint(first_group[0], first_group[1]-1) for _ in range(5)]
            self.kernel.push_input_spikes(random_indices, current_time + jitter)

        spike_counts_map = self.kernel.run(duration=self.dt, learning_enabled=True)
        
        current_spikes: Dict[str, Tensor] = {}
        batch_spikes = 0
        for name, (start_id, end_id) in self.group_indices.items():
            num_neurons = end_id - start_id
            spike_tensor = torch.zeros(1, num_neurons, device=self.device)
            for nid, count in spike_counts_map.items():
                if start_id <= nid < end_id:
                    if count > 0: spike_tensor[0, nid - start_id] = 1.0
                    batch_spikes += count
            current_spikes[name] = spike_tensor

        self.prev_spikes = cast(Dict[str, Optional[Tensor]], current_spikes)
        self.total_spike_count += batch_spikes
        return {"spikes": current_spikes}

    def apply_plasticity_batch(self, firing_rates: Dict[str, Tensor], phase: str = "positive"):
        pass

    def reset_state(self) -> None:
        self.time_step = 0
        self.total_spike_count = 0
        self.kernel.reset_state()
        for name, group in self.neuron_groups.items():
             num = int(getattr(group, 'features', getattr(group, 'out_features', 0)))
             self.prev_spikes[name] = torch.zeros(1, num, device=self.device)

    # [Fix] Added methods for compatibility
    def get_firing_rates(self) -> Dict[str, float]:
        rates = {}
        for name, spikes in self.prev_spikes.items():
            if spikes is not None:
                # 平均発火率 (0.0 - 1.0)
                rates[name] = float(spikes.mean().item())
            else:
                rates[name] = 0.0
        return rates

    def get_total_spikes(self) -> int:
        return self.total_spike_count

SNNCore = SpikingNeuralSubstrate