# snn_research/models/visual_cortex.py
# Title: Visual Cortex (Phase 23: Precision Max)
# Description: 4000次元への回帰と学習率減衰による、95%突破の最終構成

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule

class VisualCortex(nn.Module):
    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 784)
        # [TUNING] Maximize capacity safely
        self.hidden_dim = self.config.get("hidden_dim", 4000)
        self.num_layers = self.config.get("num_layers", 2)
        self.time_steps = self.config.get("time_steps", 20)
        
        self.config["tau_mem"] = 100.0 
        self.config["threshold"] = 0.5
        
        # Initial LR
        self.base_lr = 0.12 # Slightly tuned down from 0.15 for stability at 4000 dims
        self.learning_rate = self.base_lr
        
        self.ff_threshold = 2.0   
        self.input_scale = 30.0 # Increased signal
        self.input_noise_std = 0.05
        
        self.use_k_wta = True
        self.sparsity = 0.10 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self._build_architecture()
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]
        
        self.batch_count = 0

    def _build_architecture(self):
        self.substrate.add_neuron_group("Retina", self.input_dim)
        prev = "Retina"
        for i in range(self.num_layers):
            curr = f"V{i+1}"
            self.substrate.add_neuron_group(curr, self.hidden_dim)
            
            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold
            )
            name = f"{prev.lower()}_to_{curr.lower()}"
            self.substrate.add_projection(name, prev, curr, plasticity_rule=ff_rule)
            prev = curr

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device: x = x.to(self.device)
        x = x.float()
        if x.max() > 0: x = x / x.max()
        x = x * self.input_scale
        return x

    def _update_learning_rate(self):
        # Simple Step Decay: Decay every 200 batches
        decay_factor = 0.95
        if self.batch_count > 0 and self.batch_count % 200 == 0:
            self.learning_rate *= decay_factor
            # Apply to rules
            for proj in self.substrate.projections.values():
                if proj.plasticity_rule:
                    proj.plasticity_rule.base_lr = self.learning_rate

    def forward(self, x: torch.Tensor, phase: str = "wake", prepped: bool = False, update_weights: bool = True) -> Dict[str, torch.Tensor]:
        self.substrate.reset_state()
        if not prepped: x = self.prepare_input(x)

        learning_phase = "neutral"
        if phase == "wake":
            learning_phase = "positive"
            if update_weights and self.training:
                noise = torch.randn_like(x) * self.input_noise_std
                x = x + noise
        elif phase == "sleep":
            learning_phase = "negative"

        spike_sums = {name: torch.zeros(x.shape[0], int(group.features), device=self.device) 
                      for name, group in self.substrate.neuron_groups.items()}
        
        with torch.no_grad():
            for t in range(self.time_steps):
                step_in = {"Retina": x}
                out = self.substrate.forward_step(step_in, instant_plasticity=False)
                
                if self.use_k_wta:
                    self._apply_k_wta(out['spikes'])

                for name, s in out['spikes'].items():
                    if name in spike_sums:
                        spike_sums[name] += s.detach().float()
                
                del step_in, out

        mean_rates = {name: s / self.time_steps for name, s in spike_sums.items()}

        if update_weights:
            self.batch_count += 1
            self._update_learning_rate()
            
            self.substrate.apply_plasticity_batch(
                firing_rates=mean_rates,
                phase=learning_phase,
                momentum=0.9
            )

        return {
            "spikes": mean_rates, 
            "raw_spikes": {} 
        }

    def _apply_k_wta(self, activity: Dict[str, torch.Tensor]):
        k = int(self.hidden_dim * self.sparsity)
        if k <= 0: k = 1
        for name in self.layer_names:
            if name in activity:
                spikes = activity[name]
                if spikes.sum() > 0:
                    vals, _ = torch.topk(spikes, k, dim=1)
                    thresh = vals[:, -1].unsqueeze(1)
                    mask = (spikes >= thresh).float()
                    activity[name] = spikes * mask

    def get_goodness(self) -> Dict[str, float]:
        stats = {}
        for name in self.layer_names:
            spikes = self.substrate.prev_spikes.get(name)
            val = spikes.float() if spikes is not None else torch.zeros(1).to(self.device)
            stats[f"{name}_goodness"] = val.pow(2).sum(dim=1).mean().item()
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        return {"layers": {n: {"firing_rate": 0.0} for n in self.layer_names}}
    
    def reset_state(self):
        self.substrate.reset_state()