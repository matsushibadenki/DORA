# snn_research/learning_rules/forward_forward.py
# Title: FF Rule (Balanced Norm)
# Description: Peer Normを0.004に調整し、過学習と過剰抑制のバランスを取る

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from snn_research.learning_rules.base_rule import PlasticityRule


class ForwardForwardRule(PlasticityRule):
    def __init__(self, learning_rate: float = 0.10, threshold: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base_lr = learning_rate
        self.threshold = threshold
        self.step_count = 0

    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        
        phase = kwargs.get("phase", "neutral")
        if phase == "neutral":
            return None, {}

        self.step_count += 1
        current_lr = self.base_lr 

        pre_rate = pre_spikes.float()
        post_rate = post_spikes.float()
        
        goodness = post_rate.pow(2).sum(dim=1, keepdim=True)
        probs = torch.sigmoid(goodness - self.threshold)
        
        if phase == "positive":
            scale = 1.0 - probs
            direction = 1.0
        elif phase == "negative":
            scale = probs
            direction = -1.0
        else:
            return None, {}

        post_scaled = post_rate * scale 
        delta_w = torch.matmul(post_scaled.T, pre_rate) 
        
        batch_size = pre_rate.shape[0]
        if batch_size > 0:
            delta_w = (delta_w / batch_size) * current_lr * direction

        # [TUNING] Balanced Peer Norm
        # 0.002 -> 0.004 (Keep goodness in check)
        if phase == "positive":
            mean_activity = post_rate.mean(dim=0) 
            peer_penalty = 0.004 * mean_activity.unsqueeze(1) * current_weights
            delta_w -= peer_penalty

        # Minimal Weight Decay
        delta_w -= 0.00001 * current_weights 

        logs = {
            "ff_phase": phase,
            "mean_delta": delta_w.abs().mean().item(),
            "lr": current_lr
        }
        
        return delta_w, logs