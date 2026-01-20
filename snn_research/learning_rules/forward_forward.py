# ファイルパス: snn_research/learning_rules/forward_forward.py
import torch
import logging
from typing import Dict, Any, Tuple, Optional
from snn_research.learning_rules.base_rule import PlasticityRule

class ForwardForwardRule(PlasticityRule):
    def __init__(self, learning_rate: float = 0.05, threshold: float = 50.0, w_decay: float = 0.0001):
        self.lr = learning_rate
        self.threshold = threshold
        self.w_decay = w_decay

    def update(self, pre_spikes, post_spikes, current_weights, local_state=None, **kwargs):
        phase = kwargs.get("phase", "neutral")
        if phase == "neutral":
            return None, {}

        if local_state is None: local_state = {}

        # Trace for smoother activity
        trace_post = local_state.get("trace_post_rate", torch.zeros_like(post_spikes))
        alpha = 0.2
        activity = trace_post * (1 - alpha) + post_spikes * alpha
        local_state["trace_post_rate"] = activity.detach()

        # Goodness = Sum(Activity^2)
        # Using Sum to match the threshold scale
        goodness = activity.pow(2).sum(dim=1) + 1e-6 # (Batch,)

        # Probability calc
        # sigmoid( Goodness - Threshold )
        # If G > T, prob -> 1. If G < T, prob -> 0.
        logits = goodness - self.threshold
        probs = torch.sigmoid(logits)

        # Gradient Factor
        if phase == "positive":
            factor = (1.0 - probs) # Push G up
        else:
            factor = (-probs)      # Push G down

        batch_size = pre_spikes.size(0)
        weighted_activity = activity * factor.unsqueeze(1) # (Batch, N_post)

        numerator = torch.matmul(weighted_activity.t(), pre_spikes)
        delta_w = (numerator / float(batch_size)) * self.lr
        delta_w -= self.w_decay * current_weights

        return delta_w, {
            "mean_goodness": goodness.mean().item(),
            "mean_prob": probs.mean().item()
        }
    
    def get_config(self):
        return {"type": "ForwardForward", "lr": self.lr, "threshold": self.threshold}