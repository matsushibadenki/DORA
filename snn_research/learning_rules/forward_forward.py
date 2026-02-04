# snn_research/learning_rules/forward_forward.py
# Title: FF Rule (Phase 54: Freedom)
# Description: Peer Normを0.005に半減させ、ニューロンの自律的な活動を許可する

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from snn_research.learning_rules.base_rule import PlasticityRule


class ForwardForwardRule(PlasticityRule):
    def __init__(self, learning_rate: float = 0.05, threshold: float = 2.0, **kwargs: Any) -> None:
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

        # [TUNING] 0.01 -> 0.005 (Relaxed constraint)
        # Allows Goodness to grow naturally
        if phase == "positive":
            mean_activity = post_rate.mean(dim=0) 
            peer_penalty = 0.005 * mean_activity.unsqueeze(1) * current_weights
            delta_w -= peer_penalty

        # Weight Decay is still OFF
        
        logs = {
            "ff_phase": phase,
            "mean_delta": delta_w.abs().mean().item(),
            "lr": current_lr
        }
        
        return delta_w, logs