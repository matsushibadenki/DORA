# snn_research/learning_rules/forward_forward.py
# Title: Forward-Forward Rule (Final Phase 2)
# Description:
#   インプレース演算とDual-Traceを実装し、メモリ効率と学習安定性を最大化。
#   MPSのOOM問題を回避するための明示的なリソース管理を含む。

import torch
import logging
from typing import Dict, Any, Tuple, Optional
from snn_research.learning_rules.base_rule import PlasticityRule

class ForwardForwardRule(PlasticityRule):
    def __init__(self, learning_rate: float = 0.05, threshold: float = 15.0, w_decay: float = 0.0001):
        super().__init__()
        self.lr = learning_rate
        self.threshold = threshold
        self.w_decay = w_decay

    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        current_weights: torch.Tensor, 
        local_state: Optional[Dict[str, Any]] = None, 
        **kwargs
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        
        phase = kwargs.get("phase", "neutral")
        update_weights = kwargs.get("update_weights", True)

        if phase == "neutral":
            return None, {}

        if local_state is None:
            local_state = {}

        alpha = 0.2
        
        # Dual Trace Update (In-Place for Memory Efficiency)
        # Post-synaptic Trace
        if "trace_post" not in local_state:
            local_state["trace_post"] = post_spikes.detach().clone()
        else:
            local_state["trace_post"].mul_(1 - alpha).add_(post_spikes.detach(), alpha=alpha)
            
        # Pre-synaptic Trace
        if "trace_pre" not in local_state:
            local_state["trace_pre"] = pre_spikes.detach().clone()
        else:
            local_state["trace_pre"].mul_(1 - alpha).add_(pre_spikes.detach(), alpha=alpha)

        if not update_weights:
            return None, {}

        # --- Weight Update Calculation ---
        post_activity = local_state["trace_post"]
        pre_activity = local_state["trace_pre"]

        # Goodness = Sum(Activity^2)
        goodness = post_activity.pow(2).sum(dim=1) + 1e-6 

        # Probability calc
        logits = goodness - self.threshold
        probs = torch.sigmoid(logits)

        # Gradient Factor
        if phase == "positive":
            factor = (1.0 - probs)
        else:
            factor = (-probs)

        batch_size = pre_spikes.size(0)
        
        # Hebbian Term: (Post, Batch) @ (Batch, Pre) -> (Post, Pre)
        # Intermediate: (Batch, Post) * (Batch, 1)
        weighted_post = post_activity * factor.view(-1, 1)

        numerator = torch.matmul(weighted_post.t(), pre_activity)
        
        # Delta W
        delta_w = numerator.mul_(self.lr / float(batch_size))
        
        # Weight Decay (In-place)
        delta_w.sub_(current_weights, alpha=self.w_decay)

        # Metrics
        mean_goodness = goodness.mean().item()
        mean_prob = probs.mean().item()

        # [Critical] Explicit Memory Cleanup for MPS
        del weighted_post
        del numerator
        del goodness
        del probs
        del factor

        return delta_w, {
            "mean_goodness": mean_goodness,
            "mean_prob": mean_prob
        }
    
    def get_config(self):
        return {"type": "ForwardForward", "lr": self.lr, "threshold": self.threshold}