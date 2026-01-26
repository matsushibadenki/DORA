# ファイルパス: snn_research/learning_rules/stdp.py
# 日本語タイトル: Stabilized STDP with Homeostasis
# 目的・内容:
#   Objective.md Phase 2「学習安定性 > 95%」準拠の改良版学習則。
#   従来のSTDPに加え、重み正規化と恒常性維持（Homeostasis）を導入。
#   長期的な発火率の安定化を図る。

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)


class STDPRule(PlasticityRule):
    """
    Stabilized Spike-Timing-Dependent Plasticity (STDP) Rule.
    
    Improvements for Objective v2.2:
    - Weight Normalization: 重みの発散を防止
    - Homeostatic Scaling: 目標発火率からの乖離による学習率調整
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        w_max: float = 1.0,
        w_min: float = 0.0,
        target_rate: float = 0.05,  # 目標発火率 5% (Objective Axis 2)
        homeostasis_rate: float = 0.001 # 恒常性の強さ
    ):
        self.lr = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.w_max = w_max
        self.w_min = w_min
        
        self.target_rate = target_rate
        self.homeostasis_rate = homeostasis_rate

        self.A_plus = 1.0
        self.A_minus = 1.05

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_weights: torch.Tensor,
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        Args:
            pre_spikes: (Batch, N_pre)
            post_spikes: (Batch, N_post)
            current_weights: (N_post, N_pre)
        """
        if local_state is None:
            local_state = {}

        dt = kwargs.get("dt", 1.0)

        # --- 1. Trace Update ---
        trace_pre = local_state.get("trace_pre", torch.zeros_like(pre_spikes))
        decay_pre = 1.0 - (dt / self.tau_pre)
        trace_pre = trace_pre * decay_pre + pre_spikes

        trace_post = local_state.get("trace_post", torch.zeros_like(post_spikes))
        decay_post = 1.0 - (dt / self.tau_post)
        trace_post = trace_post * decay_post + post_spikes

        local_state["trace_pre"] = trace_pre.detach()
        local_state["trace_post"] = trace_post.detach()

        # --- 2. Standard STDP Calculation ---
        # LTP: Pre -> Post
        delta_w_ltp = torch.einsum("bi,bj->bij", post_spikes, trace_pre) * self.A_plus
        
        # LTD: Post -> Pre
        delta_w_ltd = torch.einsum("bi,bj->bij", trace_post, pre_spikes) * self.A_minus

        # 平均化して学習率を適用
        delta_w = (delta_w_ltp - delta_w_ltd).mean(dim=0) * self.lr

        # --- 3. Homeostatic Plasticity (Synaptic Scaling) ---
        # 長期的な発火率を推定するための移動平均（もしあれば）を利用、なければ現在のバッチの発火率
        # Objective: 発火しすぎているニューロンへの入力重みを全体的に下げる
        
        mean_firing_rate = post_spikes.mean(dim=0) # (N_post,)
        rate_diff = mean_firing_rate - self.target_rate
        
        # 発火率が高すぎる -> 重みを下げる (Negative feedback)
        # (N_post,) -> (N_post, 1) broadcast to (N_post, N_pre)
        homeostasis_term = -1.0 * rate_diff.unsqueeze(1) * current_weights * self.homeostasis_rate
        
        delta_w = delta_w + homeostasis_term

        # --- 4. Soft Bound Stabilization ---
        # 重みが上限/下限に近いほど変化量を小さくする（ソフトリミット）
        # これにより重みが極端な値に張り付くのを防ぐ
        pos_mask = delta_w > 0
        neg_mask = delta_w < 0
        
        delta_w[pos_mask] = delta_w[pos_mask] * (self.w_max - current_weights[pos_mask])
        delta_w[neg_mask] = delta_w[neg_mask] * (current_weights[neg_mask] - self.w_min)

        logs = {
            "mean_ltp": delta_w_ltp.mean().item(),
            "mean_ltd": delta_w_ltd.mean().item(),
            "mean_homeostasis": homeostasis_term.mean().item(),
            "max_trace_pre": trace_pre.max().item()
        }

        return delta_w, logs

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "StabilizedSTDP",
            "lr": self.lr,
            "tau_pre": self.tau_pre,
            "tau_post": self.tau_post,
            "target_rate": self.target_rate
        }

# --- Alias ---
STDP = STDPRule