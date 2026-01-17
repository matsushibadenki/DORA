# ファイルパス: snn_research/models/visual_cortex.py
# 日本語タイトル: Visual Cortex Model (Standard Implementation)
# 目的: Refactoring logic from run_spiking_ff_demo.py into a reusable model component

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule


class VisualCortex(nn.Module):
    """
    視覚野（Visual Cortex）モデルの標準実装。
    SpikingNeuralSubstrateを使用し、Forward-Forward学習則で構成される。

    Legacy Demo (`run_spiking_ff_demo.py`) のロジックを統合し、
    標準モデルとして再利用可能にする。

    Structure:
        - Retina (Input): 784 neurons (default)
        - V{k} (Hidden): 1500 neurons (default)
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        # Default Configuration
        self.input_dim = self.config.get("input_dim", 784)
        self.hidden_dim = self.config.get("hidden_dim", 1500)
        self.num_layers = self.config.get(
            "num_layers", 2)  # Number of hidden layers

        # SNN Configuration - Neuro-biologically plausible defaults
        self.config.setdefault("tau_mem", 0.5)      # Membrane time constant
        self.config.setdefault("threshold", 1.0)    # Firing threshold
        self.config.setdefault("dt", 1.0)           # Time step
        # Refractory period steps
        self.config.setdefault("refractory_period", 2)

        # Learning Rule Configuration
        self.learning_rate = self.config.get("learning_rate", 0.0015)
        self.ff_threshold = self.config.get("ff_threshold", 2.0)

        # Substrate Initialization
        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self._build_architecture()

        # layers attribute for Trainer compatibility
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]

        # Layer Normalization (Optional but recommended for FF stability)
        self.use_layer_norm = self.config.get("use_layer_norm", True)
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleDict()
            for name in self.layer_names:
                # Element-wise affine transformation per neuron
                self.layer_norms[name] = nn.LayerNorm(
                    self.hidden_dim, elementwise_affine=True)

    def _build_architecture(self):
        # 1. Define Neuron Groups
        # Retina (Input Layer)
        self.substrate.add_neuron_group("Retina", self.input_dim)

        # Hidden Layers (V1, V2, ...)
        prev_layer = "Retina"
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            self.substrate.add_neuron_group(layer_name, self.hidden_dim)

            # Common FF Rule for each layer
            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold
            )

            # 2. Define Projections (Feedforward)
            projection_name = f"{prev_layer.lower()}_to_{layer_name.lower()}"
            self.substrate.add_projection(
                projection_name, prev_layer, layer_name, plasticity_rule=ff_rule
            )

            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        """
        Forward pass for one time step.
        Returns:
            Dictionary of firing spikes for each layer.
        """
        # Ensure input is on correct device
        x = x.to(self.device).float()
        inputs = {"Retina": x}

        # Map generic phase to FF specific phase
        # wake -> positive (data)
        # sleep -> negative (hallucination/noise)
        learning_phase = "neutral"
        if phase == "wake":
            learning_phase = "positive"
        elif phase == "sleep":
            learning_phase = "negative"

        # Apply Layer Normalization to Membrane Potentials BEFORE Spike Generation?
        # In this architecture, we intervene in the Substrate's loop if possible,
        # but Substrate encapsulates the step.
        # Alternatively, we can normalize the *Input Current* or *Membrane Potential* if manageable.
        # Since Substrate handles internal dynamics, modifying internals is hard without changing Substrate.
        # However, for FF, normalizing the *Activity* (post-spike or potential) is key.

        # NOTE: Standard SNNs don't use LayerNorm inside the step easily.
        # But Geoffrey Hinton's FF paper explicitly normalizes activity.
        # Here we will perform a standard forward step, and IF we had access to modify dynamics, we would.
        # Instead, we will rely on PlasticityRule to normalize updates, OR
        # Customarily, we could normalize the input to the next layer?

        # Current compromise:
        # We rely on substrate.forward_step().
        # Ideally, we should inject LayerNorm into the NeuronGroup or Projection.
        # Since we can't easily do that without refactoring Substrate,
        # we will rely on the Substrate as-is for now, but if LayerNorm is crucial,
        # we might need to wrap NeuronGroups.

        # WAIT: The stability plan said "Add LayerNorm (applied to membrane potentials)".
        # To do this correctly, we might need a Custom Neuron model or
        # manually normalize after step if stateful?
        # Let's check if we can modify the mem state after the step.

        out = self.substrate.forward_step(inputs, phase=learning_phase)

        # Apply Logic: Normalize Membrane Potential for Stability
        if self.use_layer_norm:
            with torch.no_grad():
                for name, group in self.substrate.neuron_groups.items():
                    if name in self.layer_names:  # Only hidden layers
                        if hasattr(group, "mem"):
                            # Normalize Membrane Potential in-place to keep it conditioned
                            # V <- LN(V)
                            # This prevents "voltage explosion"
                            normed_mem = self.layer_norms[name](group.mem)
                            group.mem.copy_(normed_mem)

        return out

    def get_goodness(self, reduction: str = "mean") -> Dict[str, Union[float, torch.Tensor]]:
        """
        各層のGoodness（Activityの二乗平均）を返す。
        Forward-Forwardアルゴリズムにおける「エネルギー」に相当。

        Args:
            reduction: "mean" (scalar), "none" (tensor per batch), "sum" (scalar sum)
        """
        stats = {}
        # Calculate goodness for each hidden layer
        for i in range(self.num_layers):
            layer_name = f"V{i+1}"
            spikes = self.substrate.prev_spikes.get(layer_name)

            if spikes is not None:
                neuron_group = self.substrate.neuron_groups[layer_name]
                if hasattr(neuron_group, "mem"):
                    # Use Membrane Potential for Goodness (richer signal than binary spikes)
                    # V^2 mean
                    mem = neuron_group.mem  # (Batch, Features)
                    # Use absolute value or squared for energy? FF usually uses squared length of activity.
                    # Here we use squared membrane potential as proxy for "activity magnitude"
                    goodness = mem.pow(2).mean(dim=1)  # (Batch,)
                else:
                    # Fallback to spike rate if mem not available
                    goodness = spikes.float().mean(dim=1)  # (Batch,)

                # Reduction
                if reduction == "mean":
                    stats[f"{layer_name}_goodness"] = goodness.mean().item()
                    if i == 0:
                        stats["avg_pos_goodness"] = goodness.mean().item()
                elif reduction == "sum":
                    stats[f"{layer_name}_goodness"] = goodness.sum().item()
                elif reduction == "none":
                    # Tensor (Batch,)
                    stats[f"{layer_name}_goodness"] = goodness

                # Add Debug Stats if reduction is mean
                if reduction == "mean":
                    stats[f"{layer_name}_mean_mem"] = neuron_group.mem.mean().item()
                    stats[f"{layer_name}_std_mem"] = neuron_group.mem.std().item()

        return stats

    def get_state(self) -> Dict[str, Any]:
        """
        Observer用の詳細な内部状態を取得する。
        """
        state = {
            "config": self.config,
            "layers": {}
        }

        for name in self.layer_names:
            group = self.substrate.neuron_groups.get(name)
            if group:
                # Additional Stability Metrics
                weights = []
                # Find input projections to this layer
                for proj_name, proj in self.substrate.projections.items():
                    if proj_name.endswith(f"_to_{name.lower()}"):
                        weights.append(proj.synapse.weight.data)

                w_mean = 0.0
                w_std = 0.0
                if weights:
                    w_curr = weights[0]  # Primary input
                    w_mean = w_curr.mean().item()
                    w_std = w_curr.std().item()

                layer_state = {
                    # Record mean membrane potential
                    "mean_mem": group.mem.mean().item() if hasattr(group, "mem") else 0.0,
                    # Record firing rate of last step
                    "firing_rate": self.substrate.prev_spikes[name].float().mean().item() if name in self.substrate.prev_spikes else 0.0,
                    # Weight stability
                    "weight_mean": w_mean,
                    "weight_std": w_std
                }
                state["layers"][name] = layer_state

        return state

    def reset_state(self):
        """Reset internal state (membrane potentials, etc.)"""
        self.substrate.reset_state()
