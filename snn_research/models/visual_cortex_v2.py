# ファイルパス: snn_research/models/visual_cortex_v2.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, cast, List

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule
import logging

logger = logging.getLogger(__name__)

class VisualCortexV2(nn.Module):
    """
    Visual Cortex V2 - Phase 2 Rev15 (Vitality Injection)
    
    Fixes "Brain Death" (0.0 activity):
    - Inject strong BASE BIAS (+2.0) to all neurons before competition.
      This guarantees that even with zero weights, neurons will fire based on noise/input.
    - Disable Weight Decay (0.0).
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 794)
        self.hidden_dim = self.config.get("hidden_dim", 2000)
        self.num_layers = self.config.get("num_layers", 3)
        
        self.config.setdefault("dt", 1.0)
        self.config.setdefault("tau_mem", 5.0)
        self.config.setdefault("threshold", 1.0)
        
        self.learning_rate = self.config.get("learning_rate", 0.08)
        self.ff_threshold = self.config.get("ff_threshold", 2000.0) 
        self.sparsity = 0.05 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]
        
        self.label_projections = nn.ModuleDict()
        for name in self.layer_names:
            self.label_projections[name] = nn.Linear(10, self.hidden_dim, bias=False)
        
        self._build_architecture()
        
        self.activity_history: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.layer_traces: Dict[str, torch.Tensor] = {}

    def _build_architecture(self):
        self.substrate.add_neuron_group("Retina", 784)

        prev_layer = "Retina"
        for i, layer_name in enumerate(self.layer_names):
            self.substrate.add_neuron_group(layer_name, self.hidden_dim)

            # --- Rev15: No Weight Decay (0.0) ---
            # Sleep phase alone provides enough depression.
            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold,
                w_decay=0.0
            )

            projection_name = f"{prev_layer.lower()}_to_{layer_name.lower()}"
            self.substrate.add_projection(
                projection_name, prev_layer, layer_name, plasticity_rule=ff_rule
            )

            with torch.no_grad():
                proj = self.substrate.projections[projection_name]
                synapse = cast(nn.Module, proj.synapse)
                if hasattr(synapse, 'weight'):
                    w = cast(torch.Tensor, synapse.weight)
                    gain = 0.2 if i == 0 else 0.5
                    nn.init.orthogonal_(w, gain=gain)
            
            prev_layer = layer_name
            
        with torch.no_grad():
            for name, proj in self.label_projections.items():
                nn.init.orthogonal_(proj.weight, gain=5.0)

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        
        img = x[:, :784]
        lbl = x[:, 784:]
        
        img = img / (img.norm(p=2, dim=1, keepdim=True) + 1e-8) * 30.0
        
        batch_size = x.size(0)
        label_currents = {}
        for name in self.layer_names:
            label_currents[name] = self.label_projections[name](lbl)

        inputs = {"Retina": img}
        
        learning_phase = "neutral"
        inject_noise = False

        if phase == "wake":
            learning_phase = "positive"
            inject_noise = True
        elif phase == "sleep":
            learning_phase = "negative"
            inject_noise = True

        if inject_noise:
            for name in self.layer_names:
                noise = torch.randn(batch_size, self.hidden_dim, device=self.device) * 0.2
                if name not in inputs:
                    inputs[name] = noise
                else:
                    inputs[name] += noise

        simulation_steps = 6
        last_out = {}
        self.layer_traces = {name: torch.zeros(batch_size, self.hidden_dim, device=self.device) 
                             for name in self.layer_names}

        for t in range(simulation_steps):
            current_phase = learning_phase if t >= 3 else "neutral"
            out = self.substrate.forward_step(inputs, phase=current_phase)
            last_out = out

            with torch.no_grad():
                for name in self.layer_names:
                    group = self.substrate.neuron_groups[name]
                    
                    if hasattr(group, "mem"):
                        mem = cast(torch.Tensor, group.mem)
                        mem.add_(label_currents[name])

                        # --- Rev15: Vitality Injection ---
                        # Shift all potentials up by 2.0 BEFORE competition.
                        # This ensures the "winners" are always positive and active.
                        mem.add_(2.0)

                        # k-WTA
                        k = int(self.hidden_dim * self.sparsity)
                        topk_vals, _ = torch.topk(mem, k, dim=1)
                        threshold_val = topk_vals[:, -1].unsqueeze(1)
                        mask = (mem >= threshold_val).float()
                        mem.mul_(mask)
                        
                        # Bias (Post-activation support)
                        mem.add_(mask * 0.5) 

                    activity = torch.relu(mem)
                    self.layer_traces[name] = 0.8 * self.layer_traces[name] + 0.2 * activity

                    if t == simulation_steps - 1:
                        rate = (activity > 0).float().mean().item()
                        self.activity_history[name] = 0.9 * self.activity_history[name] + 0.1 * rate
        
        # Update Label Weights
        if learning_phase != "neutral" and phase != "inference":
            with torch.no_grad():
                direction = 1.0 if learning_phase == "positive" else -1.0
                lr_label = 0.05
                for name in self.layer_names:
                    v_activity = self.layer_traces[name]
                    delta_w = (lbl.t() @ v_activity) / batch_size
                    proj = self.label_projections[name]
                    proj.weight.add_(delta_w.t() * direction * lr_label)
                    # Normalize but allow growth
                    norm = proj.weight.norm(dim=1, keepdim=True) + 1e-8
                    proj.weight.div_(norm).mul_(torch.clamp(norm, max=10.0))

        return last_out

    def get_goodness(self) -> Dict[str, float]:
        stats = {}
        for name in self.layer_names:
            if name in self.layer_traces:
                trace = self.layer_traces[name]
                goodness = trace.pow(2).sum(dim=1).mean().item()
            else:
                goodness = 0.0
            stats[f"{name}_goodness"] = goodness
        return stats

    def get_stability_metrics(self) -> Dict[str, float]:
        metrics = {}
        for name in self.layer_names:
            metrics[f"{name}_firing_rate"] = self.activity_history[name]
        return metrics

    def reset_state(self):
        self.substrate.reset_state()
        self.layer_traces = {}