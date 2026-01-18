# snn_research/models/experimental/moe_model.py
import torch
import torch.nn as nn
from typing import List, Dict, Any, cast


class ExpertContainer(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model: nn.Module = model


class SpikingFrankenMoE(nn.Module):
    """
    Mixture of Experts model.
    """

    def __init__(self, experts: List[ExpertContainer], gate: nn.Module, config: Dict[str, Any], **kwargs: Any):
        super().__init__()
        self.gate = gate
        self.config = config

        self.experts = nn.ModuleList()
        for expert_container in experts:
            # Cast for mypy safety
            model_module = cast(nn.Module, expert_container.model)

            if config.get("load_checkpoint"):
                new_state_dict: Dict[str, Any] = {}
                model_module.load_state_dict(new_state_dict, strict=False)

            self.experts.append(model_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [expert(x) for expert in self.experts]
        return torch.stack(outputs).mean(dim=0)


class ContextAwareSpikingRouter(nn.Module):
    """
    Simulates context-aware routing for the demo.
    """

    def __init__(self, input_dim: int, num_experts: int, config: Dict[str, Any]):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts

        # Simple routing logic
        self.router = nn.Linear(input_dim, num_experts)
        self.context_projection = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, context: Any = None) -> torch.Tensor:
        # Base routing
        routing_logits = self.router(x)

        if context is not None and isinstance(context, torch.Tensor):
            # Ensure shape match if necessary or just add
            if context.shape[-1] == self.input_dim:
                context_bias = self.context_projection(context)
                routing_logits = routing_logits + context_bias

        return self.softmax(routing_logits)
