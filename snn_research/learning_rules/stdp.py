# snn_research/learning_rules/stdp.py
# 修正: STDPエイリアス追加、dt引数対応、kwargs対応

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class STDPRule(PlasticityRule):
    """
    Spike-Timing-Dependent Plasticity (STDP) Rule.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        a_plus: float = 1.0,
        a_minus: float = 1.2,
        w_min: float = 0.0,
        w_max: float = 1.0,
        dt: float = 1.0,
        **kwargs: Any # Legacy params (enable_homeostasis etc) absorber
    ):
        self.lr = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.dt = dt
        
        # kwargsに含まれるパラメータはログに出すか無視する
        if kwargs:
            logger.debug(f"STDPRule received extra args: {list(kwargs.keys())}")

    def update(
        self, 
        pre_spikes: torch.Tensor, 
        post_spikes: torch.Tensor, 
        current_weights: torch.Tensor, 
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        STDPによる重み更新計算。
        """
        if local_state is None:
            local_state = {}

        device = pre_spikes.device
        batch_size = pre_spikes.size(0)

        trace_pre = local_state.get("trace_pre", torch.zeros_like(pre_spikes))
        trace_post = local_state.get("trace_post", torch.zeros_like(post_spikes))

        # dtを考慮した減衰
        decay_pre = torch.exp(torch.tensor(-self.dt / self.tau_pre, device=device))
        decay_post = torch.exp(torch.tensor(-self.dt / self.tau_post, device=device))

        trace_pre = trace_pre * decay_pre + pre_spikes
        trace_post = trace_post * decay_post + post_spikes

        # 重み更新量 (Delta W) の計算
        delta_w_plus = torch.einsum("bi,bj->ij", post_spikes, trace_pre) * self.a_plus
        delta_w_minus = torch.einsum("bi,bj->ij", trace_post, pre_spikes) * self.a_minus
        
        delta_w = self.lr * (delta_w_plus - delta_w_minus) / batch_size

        local_state["trace_pre"] = trace_pre.detach()
        local_state["trace_post"] = trace_post.detach()

        logs = {
            "mean_delta_w": delta_w.mean().item(),
            "max_trace_pre": trace_pre.max().item()
        }

        return delta_w, logs

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "STDP",
            "lr": self.lr,
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post
        }

# --- Alias for Backward Compatibility ---
STDP = STDPRule