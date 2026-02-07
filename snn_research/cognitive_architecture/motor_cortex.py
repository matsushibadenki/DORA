# snn_research/cognitive_architecture/motor_cortex.py
# Title: Motor Cortex (Test Compliant)
# Description: execute_commandsとgenerate_spiking_signalメソッドを追加。

import logging
from typing import Any, Union, Dict, List, Optional
import torch

class MotorCortex:
    def __init__(self, brain=None, actuators=None, device='cpu', threshold=50.0):
        self.brain = brain
        self.actuators = actuators if actuators else []
        self.device = device
        self.threshold = threshold
        self.logger = logging.getLogger("MotorCortex")
        self.reflex_enabled = False # Default state
        print(f"🦾 [MotorCortex] Initialized. Threshold={self.threshold}")

    def monitor_and_act(self, spike_history: List[float]):
        avg_activity = sum(spike_history) / len(spike_history) if spike_history else 0
        action = "IDLE"
        reaction = "💤 Idling..."

        if avg_activity > self.threshold:
            action = "ESCAPE"
            reaction = "🏃💨 EMERGENCY EVACUATION! (Running away)"
        elif avg_activity > (self.threshold * 0.5):
            action = "ALERT"
            reaction = "👀 LOOK AROUND (Alerted)"
        
        print(f"   🧠 [MotorCortex] Activity: {avg_activity:.2f} / Thr: {self.threshold} -> Action: {action}")
        return reaction

    def generate_command(self, plan: Union[str, Dict[str, Any]]):
        if isinstance(plan, dict):
            cmd_str = str(plan.get('command', plan))
        else:
            cmd_str = str(plan)
        print(f"   🦾 [MotorCortex] Executing Plan: {cmd_str}")

    # [Fix] Added for test compatibility
    def execute_commands(self, commands: List[Dict[str, Any]]) -> List[str]:
        """
        Execute a sequence of commands.
        """
        logs = []
        for cmd in commands:
            self.generate_command(cmd)
            logs.append(f"Executed: {cmd}")
        return logs

    # [Fix] Added for reflex testing
    def generate_spiking_signal(self, sensory_input: Any) -> Optional[int]:
        """
        Generate a reflex signal based on sensory input.
        Returns 1 (spike) or 0 (no spike), or None if no reflex.
        """
        if not self.reflex_enabled:
            return None
            
        signal_val = 0.0
        if isinstance(sensory_input, torch.Tensor):
            signal_val = sensory_input.mean().item()
        elif isinstance(sensory_input, (float, int)):
            signal_val = float(sensory_input)
            
        # Simple reflex logic
        if abs(signal_val) > 0.5: # Arbitrary reflex threshold
            return 1
        return 0

    def _trigger_reflex(self, action_type: str):
        pass