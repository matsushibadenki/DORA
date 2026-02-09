# directory: snn_research/core
# file: snn_core.py
# title: Spiking Neural Substrate v3.13 (OS Compatible)
# description: ニューラルモルフィックOSとの互換性レイヤー(wake_up, process_step等)を追加し、SpikingNeuralSubstrateとして定義。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class SpikingNeuralSubstrate(nn.Module):
    """
    Spiking Neural Substrate (SNN Core).
    Standard LIF-based SNN implementation compatible with Neuromorphic OS.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, config: Dict[str, Any] = None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.config = config or {}
        
        # Layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Parameters
        self.threshold = self.config.get("threshold", 1.0)
        self.decay = self.config.get("decay", 0.5)
        
        self.act = nn.ReLU() # Simplified activation for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. 
        Args:
            x: (Batch, Input) or (Time, Batch, Input)
        Returns:
            output: (Batch, Output) or (Time, Batch, Output)
        """
        # Case 1: (Batch, Input)
        if x.dim() == 2:
            return self._step(x)
            
        # Case 2: (Time, Batch, Input)
        elif x.dim() == 3:
            time_steps = x.size(0)
            outputs = []
            for t in range(time_steps):
                out = self._step(x[t])
                outputs.append(out)
            return torch.stack(outputs, dim=0)
            
        else:
            # Fallback for unexpected shapes
            if x.shape[-1] == self.input_size:
                return self._step(x)
            raise ValueError(f"Unsupported input shape: {x.shape}")

    def _step(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_layer(x))
        h = self.act(self.hidden_layer(h))
        return self.output_layer(h)

    # --- OS Compatibility Layer ---
    def wake_up(self):
        """OS wake up signal"""
        pass

    def sleep(self):
        """OS sleep signal"""
        pass
    
    def process_step(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """OS process step"""
        return self.forward(sensory_input)

# Alias for backward compatibility and simpler naming
SNNCore = SpikingNeuralSubstrate