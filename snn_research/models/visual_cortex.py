# ファイルパス: snn_research/models/visual_cortex.py
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, cast, List

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule


class VisualCortex(nn.Module):
    """
    Visual Cortex - "Defibrillator" Edition
    強制的に発火を引き起こすためのノイズ注入と高ゲイン設定を適用。
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 784)
        self.hidden_dim = self.config.get("hidden_dim", 1500)
        self.num_layers = self.config.get("num_layers", 2)

        # --- Aggressive Tuning for Kickstart ---
        self.config.setdefault("tau_mem", 20.0)
        # Lower threshold significantly
        self.config.setdefault("threshold", 0.5)
        self.config.setdefault("dt", 1.0)
        self.config.setdefault("refractory_period", 2)

        # High Learning Rate for initial plasticity
        self.learning_rate = self.config.get("learning_rate", 0.08)
        self.ff_threshold = self.config.get("ff_threshold", 2.0)

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self._build_architecture()

        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]

        # LayerNorm
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

            # --- High Variance Initialization ---
            with torch.no_grad():
                proj = self.substrate.projections[projection_name]
                # Normal distribution with larger std to ensure some weights are strong enough
                nn.init.normal_(proj.synapse.weight, mean=0.0, std=0.5)

            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()

        # --- 1. Super Boost Input ---
        # Normalize and then scale massively to ensure spikes at Retina
        x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8) * 50.0

        inputs = {"Retina": x}

        learning_phase = "neutral"
        if phase == "wake":
            learning_phase = "positive"
        elif phase == "sleep":
            learning_phase = "negative"

        # --- 2. Inject Background Noise (Spontaneous Activity) ---
        # "Brain is never silent."
        # 各レイヤーにランダムな電流を注入し、強制的に発火機会を作る
        batch_size = x.size(0)
        noise_level = 1.5  # 閾値(0.5)を超える可能性が高いレベル

        for name in self.layer_names:
            noise = torch.randn(batch_size, self.hidden_dim,
                                device=self.device) * noise_level
            if name not in inputs:
                inputs[name] = noise
            else:
                inputs[name] += noise

        out = self.substrate.forward_step(inputs, phase=learning_phase)

        # --- 3. LayerNorm Stabilization ---
        if self.use_layer_norm:
            with torch.no_grad():
                for name in self.layer_names:
                    group = self.substrate.neuron_groups[name]
                    if hasattr(group, "mem"):
                        current_mem = group.mem
                        normed_mem = self.layer_norms[name](current_mem)
                        # バイアスを足して発火しやすく維持
                        group.mem.copy_(normed_mem + 0.2)

        return out

    def get_goodness(self, reduction: str = "mean") -> Dict[str, Union[float, torch.Tensor]]:
        stats: Dict[str, Union[float, torch.Tensor]] = {}
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            group = self.substrate.neuron_groups[layer_name]

            if hasattr(group, "mem"):
                mem = cast(torch.Tensor, group.mem)
                goodness = mem.pow(2).mean(dim=1)
            else:
                spikes = self.substrate.prev_spikes.get(layer_name)
                # Fallback: if no spikes, goodness is 0, which is bad.
                # Add small epsilon to avoid total silence in stats
                val = spikes.float().mean(
                    dim=1) if spikes is not None else torch.zeros(1).to(self.device)
                goodness = val + 1e-6

            if reduction == "mean":
                stats[f"{layer_name}_goodness"] = goodness.mean().item()
            elif reduction == "none":
                stats[f"{layer_name}_goodness"] = goodness
        return stats

    def get_state(self) -> Dict[str, Any]:
        # (変更なし)
        state = {"config": self.config, "layers": {}}
        for name in self.layer_names:
            group = self.substrate.neuron_groups[name]
            spikes = self.substrate.prev_spikes.get(name)
            firing_rate = spikes.float().mean().item() if spikes is not None else 0.0

            weights = []
            for proj_name, proj in self.substrate.projections.items():
                if proj_name.endswith(f"_to_{name.lower()}"):
                    weights.append(proj.synapse.weight.data)

            w_mean = weights[0].mean().item() if weights else 0.0

            state["layers"][name] = {
                "mean_mem": cast(torch.Tensor, group.mem).mean().item() if hasattr(group, "mem") else 0.0,
                "firing_rate": firing_rate,
                "weight_mean": w_mean
            }
        return state

    def reset_state(self):
        self.substrate.reset_state()
