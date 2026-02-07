# snn_research/models/visual_cortex.py
# Title: Visual Cortex (Phase 76: Negative Hunter) - Mypy Fixed
# Description: self.substrate を Any にキャストし、Tensor判定エラーを回避。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, cast

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule

class VisualCortex(nn.Module):
    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 784)
        self.hidden_dim = self.config.get("hidden_dim", 4000)
        self.num_layers = self.config.get("num_layers", 2)
        self.time_steps = self.config.get("time_steps", 30)
        
        self.config["tau_mem"] = 100.0 
        self.threshold = 0.5
        self.config["threshold"] = self.threshold
        self.safety_limit = 40.0
        self.homeostasis_rate = 0.005
        self.base_lr = 0.05 
        self.learning_rate = self.base_lr
        self.ff_threshold = 2.0   
        self.input_scale = 16.0 
        self.input_noise_std = 0.08
        self.use_k_wta = True
        self.sparsity = 0.08 

        # [Fix] Cast to Any to prevent static analysis errors on dynamic attributes
        self.substrate: Any = SpikingNeuralSubstrate(self.config, self.device)
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
        decay_step = 300
        decay_rate = 0.99
        if self.batch_count > 0 and self.batch_count % decay_step == 0:
            self.learning_rate *= decay_rate
            for proj in self.substrate.projections.values():
                if proj.plasticity_rule:
                    proj.plasticity_rule.base_lr = self.learning_rate

    def _apply_guardrail_homeostasis(self, current_goodness: float):
        if current_goodness > self.safety_limit:
            diff = current_goodness - self.safety_limit
            adjustment = self.homeostasis_rate * diff
            adjustment = min(adjustment, 0.05)
            
            self.threshold += adjustment
            if self.threshold > 5.0: self.threshold = 5.0
            
            self.substrate.config["threshold"] = self.threshold
            for group in self.substrate.neuron_groups.values():
                if hasattr(group, 'v_threshold'):
                    setattr(group, 'v_threshold', self.threshold)

    def forward(self, x: torch.Tensor, phase: str = "wake", prepped: bool = False, update_weights: bool = True) -> Dict[str, Any]:
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

        spike_sums = {}
        for name, group in self.substrate.neuron_groups.items():
            features = int(getattr(group, 'features', getattr(group, 'out_features', 0)))
            if features == 0 and isinstance(group, dict): # For dict compatibility
                features = int(group.get('size', 0))
            spike_sums[name] = torch.zeros(x.shape[0], features, device=self.device)
        
        with torch.no_grad():
            for t in range(self.time_steps):
                step_in = {"Retina": x}
                out = self.substrate.forward_step(step_in, instant_plasticity=False)
                
                if self.use_k_wta:
                    self._apply_k_wta(cast(Dict[str, torch.Tensor], out['spikes']))

                for name, s in out['spikes'].items():
                    if name in spike_sums and isinstance(s, torch.Tensor):
                        spike_sums[name] += s.detach().float()
                
                del step_in, out

        mean_rates = {name: s / self.time_steps for name, s in spike_sums.items()}
        
        v1_spikes = self.substrate.prev_spikes.get("V1")
        if v1_spikes is not None:
            current_goodness = v1_spikes.float().pow(2).sum(dim=1).mean().item()
        else:
            current_goodness = 0.0

        if update_weights:
            self.batch_count += 1
            self._update_learning_rate()
            if phase == "wake":
                self._apply_guardrail_homeostasis(current_goodness)

            self.substrate.apply_plasticity_batch(
                firing_rates=mean_rates,
                phase=learning_phase
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