# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: Spiking Neural Substrate (Fix: LIFNeuron Args)
# 修正内容: 
#   - LIFNeuron初期化時の不要な v_reset 引数を削除
#   - ニューロンインスタンスの set_stateful(True) を呼び出し、膜電位を維持するように設定

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

# Import Native LIFNeuron
try:
    from snn_research.core.neurons.lif_neuron import LIFNeuron
except ImportError:
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from snn_research.core.neurons.lif_neuron import LIFNeuron

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class SynapticProjection(nn.Module):
    """
    ニューロン集団間のシナプス結合を表現するクラス。
    """
    def __init__(self, in_features: int, out_features: int, plasticity_rule: Optional[PlasticityRule] = None):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.plasticity_rule = plasticity_rule
        
        nn.init.orthogonal_(self.synapse.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.synapse(x)
    
    def apply_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, **kwargs) -> Dict[str, Any]:
        if self.plasticity_rule is None:
            return {}
            
        with torch.no_grad():
            delta_w, logs = self.plasticity_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                current_weights=self.synapse.weight.data,
                **kwargs
            )
            
            if delta_w is not None:
                self.synapse.weight.data += delta_w
                self.synapse.weight.data.clamp_(-5.0, 5.0)
                
        return logs

class SpikingNeuralSubstrate(nn.Module):
    """
    Neuromorphic OSのための汎用神経基盤（Kernel）。
    """
    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        
        self.time_step = 0
        self.dt = config.get("dt", 1.0)
        
        self.neuron_groups: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.topology: List[Dict[str, str]] = []
        self.prev_spikes: Dict[str, torch.Tensor] = {}

        logger.info("⚡ SpikingNeuralSubstrate initialized.")

    def add_neuron_group(self, name: str, num_neurons: int, neuron_model: Optional[nn.Module] = None):
        """ニューロン集団（領野）を追加"""
        if neuron_model is None:
            # 修正: v_reset 引数を削除
            neuron_model = LIFNeuron(
                features=num_neurons,
                tau_mem=self.config.get("tau_mem", 20.0),
                v_threshold=self.config.get("threshold", 1.0),
                dt=self.dt
            )
        
        # 修正: ニューロンが状態を持てるように設定
        if hasattr(neuron_model, "set_stateful"):
            neuron_model.set_stateful(True) # type: ignore
        
        self.neuron_groups[name] = neuron_model.to(self.device)
        self.prev_spikes[name] = None 
        
        logger.debug(f"  + Group added: {name} ({num_neurons} neurons)")

    def add_projection(self, name: str, source: str, target: str, plasticity_rule: Optional[PlasticityRule] = None):
        if source not in self.neuron_groups or target not in self.neuron_groups:
            raise ValueError(f"Source {source} or Target {target} not found.")
            
        src_dim = self.neuron_groups[source].features # type: ignore
        tgt_dim = self.neuron_groups[target].features # type: ignore
        
        projection = SynapticProjection(src_dim, tgt_dim, plasticity_rule)
        self.projections[name] = projection.to(self.device)
        
        self.topology.append({"name": name, "src": source, "tgt": target})
        logger.debug(f"  + Projection added: {name} ({source} -> {target})")

    def forward_step(self, external_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        self.time_step += 1
        
        batch_size = 1
        for inp in external_inputs.values():
            batch_size = inp.shape[0]
            break
            
        for name, group in self.neuron_groups.items():
            if self.prev_spikes[name] is None or self.prev_spikes[name].shape[0] != batch_size: # type: ignore
                num_neurons = group.features # type: ignore
                self.prev_spikes[name] = torch.zeros(batch_size, num_neurons, device=self.device)

        # 1. Integration
        current_inputs: Dict[str, torch.Tensor] = {}
        
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']
            
            proj_module = self.projections[proj_name]
            src_spikes_prev = self.prev_spikes[src_name]
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
                    current_inputs[group_name] = current_inputs[group_name] + inp
        
        # 2. Dynamics
        current_spikes: Dict[str, torch.Tensor] = {}
        
        for name, group in self.neuron_groups.items():
            if name in current_inputs:
                input_current = current_inputs[name]
            else:
                num_neurons = group.features # type: ignore
                input_current = torch.zeros(batch_size, num_neurons, device=self.device)
            
            spikes, _ = group(input_current)
            current_spikes[name] = spikes
            
        # 3. Plasticity
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']
            
            proj_module = self.projections[proj_name] # type: ignore
            
            if hasattr(proj_module, 'plasticity_rule') and proj_module.plasticity_rule is not None:
                src_spikes_prev = self.prev_spikes[src_name]
                tgt_spikes_curr = current_spikes[tgt_name]
                
                proj_module.apply_plasticity(
                    pre_spikes=src_spikes_prev,
                    post_spikes=tgt_spikes_curr
                )

        # 4. Update State
        self.prev_spikes = current_spikes
        
        return {
            "spikes": current_spikes
        }

    def reset_state(self):
        self.time_step = 0
        self.prev_spikes = {}
        
        for name, group in self.neuron_groups.items():
            if hasattr(group, 'reset'):
                group.reset()
            self.prev_spikes[name] = None
            
        logger.info("🔄 Substrate state reset.")

# --- Alias for Backward Compatibility ---
SNNCore = SpikingNeuralSubstrate