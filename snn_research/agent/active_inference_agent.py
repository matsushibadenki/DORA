# snn_research/agent/active_inference_agent.py
# Title: Active Inference Agent (Mypy Fix)
# Description: snn_coreへのキャストを追加し、"Tensor not callable" エラーを解消。

from typing import Dict, Any, cast
import torch
from snn_research.agent.base_agent import BaseAgent
from snn_research.core.snn_core import SpikingNeuralSubstrate, SNNCore # Fix import

class ActiveInferenceAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.belief_state = torch.zeros(10)
        # Assuming snn_core is initialized elsewhere or passed
        self.snn_core: Any = None 

    def step(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform active inference step:
        1. Perception (update internal state)
        2. Action selection (minimize expected free energy)
        """
        if self.snn_core is not None:
            # [Fix] Cast to specific class to access methods
            core = cast(SpikingNeuralSubstrate, self.snn_core)
            firing_rates = core.get_firing_rates()
            # Do something with firing rates
            pass
            
        action = {"action": "explore", "value": 0.5}
        return action

    def update_beliefs(self, observation: torch.Tensor):
        pass