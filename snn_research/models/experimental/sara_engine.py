# directory: snn_research/models/experimental
# file: sara_engine.py
# title: SARA Engine v10.0 (Integrated Autonomous Intelligence Core)
# description: 予測符号化、能動的推論、物理法則制約、メタ認知、および記憶管理を統合した自律学習エンジンの最上位実装。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List

from snn_research.core.neurons.lif_neuron import LIFNeuron
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator

class SARAMemory(nn.Module):
    """
    SARA Episodic/Working Memory Module.
    Stores high-level state representations for retrieval and consolidation.
    """
    def __init__(self, capacity: int = 1000, state_dim: int = 128):
        super().__init__()
        self.capacity = capacity
        self.state_dim = state_dim
        self.register_buffer('memory_states', torch.zeros(capacity, state_dim))
        self.register_buffer('memory_values', torch.zeros(capacity, 1))
        self.register_buffer('write_pointer', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_full', torch.zeros(1, dtype=torch.bool))

    def store(self, state: torch.Tensor, value: float = 1.0):
        if state.dim() > 1:
            state = state.mean(dim=0)
        if state.shape[-1] != self.state_dim:
             return
        idx = self.write_pointer.item()
        self.memory_states[idx] = state.detach()
        self.memory_values[idx] = value
        self.write_pointer.fill_((idx + 1) % self.capacity)
        if (idx + 1) == self.capacity:
            self.is_full.fill_(True)

    def retrieve(self, query: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if query.shape[-1] != self.state_dim:
             batch_size = query.shape[0]
             device = query.device
             return torch.zeros(batch_size, k, self.state_dim, device=device), torch.zeros(batch_size, k, device=device)
        mem_norm = F.normalize(self.memory_states, p=2, dim=1)
        query_norm = F.normalize(query, p=2, dim=1)
        scores = torch.mm(query_norm, mem_norm.t())
        top_k_scores, indices = torch.topk(scores, k, dim=1)
        retrieved_states = self.memory_states[indices]
        return retrieved_states, top_k_scores

    def get_all(self):
        limit = self.capacity if self.is_full.item() else self.write_pointer.item()
        return self.memory_states[:limit]
        
    def forward(self, x):
        return self.retrieve(x)

class MetaCognitiveMonitor(nn.Module):
    def __init__(self, history_size: int = 100):
        super().__init__()
        self.history_size = history_size
        self.register_buffer('error_history', torch.zeros(history_size))
        self.register_buffer('pointer', torch.zeros(1, dtype=torch.long))

    def update(self, current_error: float) -> float:
        self.error_history[self.pointer] = current_error
        self.pointer = (self.pointer + 1) % self.history_size
        short_term = self.error_history[-10:].mean()
        long_term = self.error_history.mean()
        surprise_factor = F.relu(short_term - long_term)
        return surprise_factor.item()

class SARAEngine(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 action_dim: int,
                 config: Dict[str, Any] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        self.perception_core = nn.LSTMCell(input_dim + action_dim, hidden_dim)
        self.sensory_predictor = nn.Linear(hidden_dim, input_dim)
        self.state_predictor = nn.Linear(hidden_dim, hidden_dim)
        self.action_generator = nn.Linear(hidden_dim, action_dim)
        
        self.meta_monitor = MetaCognitiveMonitor()
        self.physics_evaluator = PhysicsEvaluator(self.config.get("physics", {}))
        self.memory = SARAMemory(capacity=1000, state_dim=hidden_dim)
        
        self.plasticity_level = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.energy_reserve = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, 
                sensory_input: torch.Tensor, 
                prev_action: torch.Tensor,
                prev_state: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        
        batch_size = sensory_input.size(0)
        hx, cx = prev_state
        
        # Perception
        combined_input = torch.cat([sensory_input, prev_action], dim=1)
        next_hx, next_cx = self.perception_core(combined_input, (hx, cx))
        
        # Prediction
        pred_sensory = self.sensory_predictor(next_hx)
        pred_state = self.state_predictor(next_hx)
        
        # Error
        sensory_error = sensory_input - pred_sensory
        error_magnitude = torch.norm(sensory_error, dim=1).mean()
        
        # Meta-Cognition
        surprise_score = self.meta_monitor.update(error_magnitude.item())
        self.plasticity_level.data = torch.clamp(
            self.plasticity_level + 0.01 * (surprise_score - 0.5), 
            0.01, 1.0
        )
        
        # Physics Check (Batch,)
        physics_consistency = self.physics_evaluator.evaluate_state_consistency(
            next_hx.unsqueeze(1)
        )
        
        # Action Generation
        base_action = self.action_generator(next_hx)
        
        # Action Modulation via Physics Consistency (Vectorized)
        consistency_factor = physics_consistency.unsqueeze(1) # (Batch, 1)
        scale = torch.where(
            consistency_factor < 0.5, 
            consistency_factor, 
            torch.ones_like(consistency_factor)
        )
        action = base_action * scale

        # Memory
        if surprise_score > 0.8:
            self.memory.store(next_hx, value=surprise_score)

        # Homeostasis
        energy_cost = 0.001 * error_magnitude + 0.0001 * torch.norm(action)
        self.energy_reserve.data = torch.clamp(self.energy_reserve - energy_cost, 0.0, 1.0)

        return {
            "action": torch.tanh(action),
            "next_state": (next_hx, next_cx),
            "pred_sensory": pred_sensory,
            "sensory_error": sensory_error,
            "loss_components": {
                "prediction_error": error_magnitude,
                "physics_penalty": 1.0 - physics_consistency.mean(),
                "energy_cost": energy_cost
            },
            "meta_info": {
                "plasticity": self.plasticity_level.item(),
                "surprise": surprise_score,
                "energy": self.energy_reserve.item()
            }
        }

    def get_initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def reset_homeostasis(self):
        self.plasticity_level.data.fill_(0.1)
        self.energy_reserve.data.fill_(1.0)