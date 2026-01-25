# ファイルパス: snn_research/models/visual_cortex.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, cast, List

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule


class VisualCortex(nn.Module):
    """Visual Cortex - 'Defibrillator' Edition (Type Safe)"""

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 784)
        self.hidden_dim = self.config.get("hidden_dim", 1500)
        self.num_layers = self.config.get("num_layers", 2)

        self.config.setdefault("tau_mem", 20.0)
        self.config.setdefault("threshold", 0.5)
        self.config.setdefault("dt", 1.0)
        self.config.setdefault("refractory_period", 2)

        self.learning_rate = self.config.get("learning_rate", 0.08)
        self.ff_threshold = self.config.get("ff_threshold", 2.0)

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self._build_architecture()

        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]

        self.use_layer_norm = self.config.get("use_layer_norm", True)
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleDict()
            for name in self.layer_names:
                self.layer_norms[name] = nn.LayerNorm(
                    self.hidden_dim, elementwise_affine=True)

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

            with torch.no_grad():
                proj = self.substrate.projections[projection_name]
                synapse = cast(nn.Module, proj.synapse)
                if hasattr(synapse, 'weight'):
                    # Safe initialization
                    w = cast(torch.Tensor, synapse.weight)
                    nn.init.normal_(w, mean=0.0, std=0.5)

            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8) * 50.0

        inputs = {"Retina": x}

        learning_phase = "neutral"
        if phase == "wake":
            learning_phase = "positive"
        elif phase == "sleep":
            learning_phase = "negative"

        batch_size = x.size(0)
        noise_level = 1.5

        for name in self.layer_names:
            noise = torch.randn(batch_size, self.hidden_dim,
                                device=self.device) * noise_level
            if name not in inputs:
                inputs[name] = noise
            else:
                inputs[name] += noise

        out = self.substrate.forward_step(inputs, phase=learning_phase)

        if self.use_layer_norm:
            with torch.no_grad():
                for name in self.layer_names:
                    group = self.substrate.neuron_groups[name]
                    if hasattr(group, "mem"):
                        mem = cast(torch.Tensor, group.mem)
                        normed_mem = self.layer_norms[name](mem)
                        mem.copy_(normed_mem + 0.2)

        return out

    def get_goodness(self, reduction: str = "mean") -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            group = self.substrate.neuron_groups[layer_name]

            if hasattr(group, "mem"):
                mem = cast(torch.Tensor, group.mem)
                goodness = mem.pow(2).mean(dim=1)
            else:
                spikes = self.substrate.prev_spikes.get(layer_name)
                val = spikes.float().mean(
                    dim=1) if spikes is not None else torch.zeros(1).to(self.device)
                goodness = val + 1e-6

            if reduction == "mean":
                stats[f"{layer_name}_goodness"] = goodness.mean().item()
            elif reduction == "none":
                stats[f"{layer_name}_goodness"] = goodness
        return stats

    def get_state(self) -> Dict[str, Any]:
        state = {"config": self.config, "layers": {}}
        for name in self.layer_names:
            group = self.substrate.neuron_groups[name]
            spikes = self.substrate.prev_spikes.get(name)
            firing_rate = spikes.float().mean().item() if spikes is not None else 0.0

            # Fix: Explicitly type the list to avoid Union[Tensor, Module] inference
            weights: List[torch.Tensor] = []
            for proj_name, proj in self.substrate.projections.items():
                if proj_name.endswith(f"_to_{name.lower()}"):
                    synapse = cast(nn.Module, proj.synapse)
                    if hasattr(synapse, 'weight'):
                        # Cast strictly to Tensor before appending
                        w = cast(torch.Tensor, synapse.weight)
                        weights.append(w.data)

            w_mean = 0.0
            if weights:
                # Cast implicitly handled by list type, but explicit cast here for safety if needed
                w_mean = torch.mean(weights[0]).item()

            state["layers"][name] = {
                "mean_mem": cast(torch.Tensor, group.mem).mean().item() if hasattr(group, "mem") else 0.0,
                "firing_rate": firing_rate,
                "weight_mean": w_mean
            }
        return state

    def reset_state(self):
        self.substrate.reset_state()