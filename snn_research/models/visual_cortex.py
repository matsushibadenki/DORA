import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, cast

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule

class VisualCortex(nn.Module):
    """
    Visual Cortex - Corrected Scale & State Tracking Edition
    Goodness計算をSumベースに変更し、ベンチマークに必要な状態監視機能を復元。
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 784)
        self.hidden_dim = self.config.get("hidden_dim", 1500)
        self.num_layers = self.config.get("num_layers", 2)

        # Standard SNN Parameters
        self.config.setdefault("tau_mem", 20.0)      
        self.config.setdefault("threshold", 1.0)    
        self.config.setdefault("dt", 1.0)           

        # FF Parameters
        self.learning_rate = self.config.get("learning_rate", 0.05)
        self.ff_threshold = self.config.get("ff_threshold", 50.0) 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self._build_architecture()
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]

        # LayerNorm (Stability)
        self.use_layer_norm = self.config.get("use_layer_norm", True)
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleDict()
            for name in self.layer_names:
                self.layer_norms[name] = nn.LayerNorm(self.hidden_dim)

    def _build_architecture(self):
        self.substrate.add_neuron_group("Retina", self.input_dim)
        prev_layer = "Retina"
        
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            self.substrate.add_neuron_group(layer_name, self.hidden_dim)

            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold
            )

            projection_name = f"{prev_layer.lower()}_to_{layer_name.lower()}"
            self.substrate.add_projection(
                projection_name, prev_layer, layer_name, plasticity_rule=ff_rule
            )
            
            # Robust Initialization
            with torch.no_grad():
                proj = self.substrate.projections[projection_name]
                nn.init.kaiming_uniform_(proj.synapse.weight, a=0.1)

            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        
        # Simple robust scaling (Input is 0-1, spikes need current ~1.0+)
        inputs = {"Retina": x * 2.5}

        learning_phase = "neutral"
        if phase == "wake":
            learning_phase = "positive"
        elif phase == "sleep":
            learning_phase = "negative"
        
        out = self.substrate.forward_step(inputs, phase=learning_phase)
        
        # Apply LayerNorm to Membrane Potentials for stability
        if self.use_layer_norm:
            with torch.no_grad():
                for name in self.layer_names:
                    group = self.substrate.neuron_groups[name]
                    if hasattr(group, "mem"):
                        # Normalize to mean 0, std 1
                        normed = self.layer_norms[name](group.mem)
                        # Shift back to allow silence (negative) and firing (positive)
                        group.mem.copy_(normed)

        return out

    def get_goodness(self, reduction: str = "mean") -> Dict[str, Union[float, torch.Tensor]]:
        stats = {}
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            group = self.substrate.neuron_groups[layer_name]
            
            # Goodness = Sum of Squared Activity
            if hasattr(group, "mem"):
                mem = cast(torch.Tensor, group.mem)
                activity = torch.relu(mem)
                goodness = activity.pow(2).sum(dim=1) # (Batch,)
            else:
                spikes = self.substrate.prev_spikes.get(layer_name)
                val = spikes.float().sum(dim=1) if spikes is not None else torch.zeros(1).to(self.device)
                goodness = val

            if reduction == "mean":
                stats[f"{layer_name}_goodness"] = goodness.mean().item()
            elif reduction == "none":
                stats[f"{layer_name}_goodness"] = goodness
        return stats

    def get_state(self) -> Dict[str, Any]:
        """
        ベンチマークや可視化に必要な内部状態を返す
        """
        state = {"config": self.config, "layers": {}}
        for name in self.layer_names:
            group = self.substrate.neuron_groups[name]
            spikes = self.substrate.prev_spikes.get(name)
            firing_rate = spikes.float().mean().item() if spikes is not None else 0.0
            
            # --- 復元した箇所 ---
            # 1. 重み統計 (Weight Stats)
            weights = []
            for proj_name, proj in self.substrate.projections.items():
                if proj_name.endswith(f"_to_{name.lower()}"):
                    weights.append(proj.synapse.weight.data)
            
            w_mean = weights[0].mean().item() if weights else 0.0
            
            # 2. 膜電位統計 (Membrane Potential Stats)
            mean_mem = 0.0
            if hasattr(group, "mem"):
                 mean_mem = cast(torch.Tensor, group.mem).mean().item()
            # ------------------------

            state["layers"][name] = {
                "firing_rate": firing_rate,
                "mean_mem": mean_mem,      # これが必要です
                "weight_mean": w_mean      # これも必要です
            }
        return state
    
    def reset_state(self):
        self.substrate.reset_state()