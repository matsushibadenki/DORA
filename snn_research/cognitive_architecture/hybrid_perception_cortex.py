# ファイルパス: snn_research/cognitive_architecture/hybrid_perception_cortex.py
# (Phase 3: Cortical Column Integrated - Type Safe)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .som_feature_map import SomFeatureMap
from .global_workspace import GlobalWorkspace
from snn_research.core.cortical_column import CorticalColumn

class HybridPerceptionCortex(nn.Module):
    column: CorticalColumn
    prev_column_state: Optional[Dict[str, torch.Tensor]]

    def __init__(
        self,
        workspace: GlobalWorkspace,
        num_neurons: int,
        feature_dim: int = 64,
        som_map_size=(8, 8),
        stdp_params: Optional[Dict[str, Any]] = None,
        cortical_column: Optional[CorticalColumn] = None
    ):
        super().__init__()
        self.workspace = workspace
        self.num_neurons = num_neurons
        self.feature_dim = feature_dim

        if cortical_column is None:
            self.column = CorticalColumn(
                input_dim=num_neurons,
                column_dim=feature_dim,
                output_dim=feature_dim,
                neuron_config={'type': 'lif', 'tau_mem': 20.0, 'base_threshold': 1.0}
            )
        else:
            self.column = cortical_column

        self.input_projection = nn.Linear(feature_dim, feature_dim)

        if stdp_params is None:
            stdp_params = {'learning_rate': 0.005, 'a_plus': 1.0, 'a_minus': 1.0, 'tau_trace': 20.0}

        som_total_neurons = som_map_size[0] * som_map_size[1]
        
        self.som = SomFeatureMap(
            input_dim=feature_dim,
            num_neurons=som_total_neurons,
            map_size=som_map_size,
            stdp_params=stdp_params
        )

        self.prev_column_state = None
        print("🧠 ハイブリッド知覚野 (Cortical Column + SOM) が初期化されました。")

    def forward(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        return self.perceive(sensory_input)

    def perceive(self, sensory_input: torch.Tensor) -> Dict[str, Any]:
        if sensory_input.device != next(self.parameters()).device:
            sensory_input = sensory_input.to(next(self.parameters()).device)

        if sensory_input.dim() == 2:
            input_signal = sensory_input.float().mean(dim=0).unsqueeze(0)
        else:
            input_signal = sensory_input.float().unsqueeze(0)

        out_ff, out_fb, current_states = self.column(
            input_signal, self.prev_column_state)

        self.prev_column_state = {k: v.detach() for k, v in current_states.items()}

        column_output = out_ff.squeeze(0)
        feature_vector = torch.relu(self.input_projection(column_output))

        if feature_vector.dim() > 1:
             feature_vector = feature_vector.mean(dim=0)
             
        feature_vector_flat = feature_vector.view(-1)
        
        for _ in range(1):
            som_spikes = self.som(feature_vector_flat.unsqueeze(0))
            if hasattr(self.som, 'update_weights'):
                self.som.update_weights(feature_vector_flat.unsqueeze(0), som_spikes)

        final_som_activation = self.som(feature_vector_flat.unsqueeze(0))

        column_activity = sum(t.mean().item() for t in current_states.values()) / len(current_states)

        return {
            "features": final_som_activation,
            "column_activity": column_activity,
            "type": "perception",
            "details": f"Processed via Cortical Column (Activity: {column_activity:.2f})"
        }

    def perceive_and_upload(self, spike_pattern: torch.Tensor) -> None:
        result = self.perceive(spike_pattern)
        input_strength = spike_pattern.float().mean().item()
        salience = min(1.0, (result["column_activity"] + input_strength) * 5.0)

        perception_data = {
            "type": "perception",
            "features": result["features"],
            "details": result["details"]
        }

        self.workspace.upload_to_workspace(
            source_name="perception",
            content=perception_data,
            salience=salience
        )