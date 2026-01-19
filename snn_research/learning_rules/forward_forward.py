# ファイルパス: snn_research/learning_rules/forward_forward.py
import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)


class ForwardForwardRule(PlasticityRule):
    """
    Forward-Forward Learning Rule (Robust SNN Version).
    """

    def __init__(
        self,
        learning_rate: float = 0.05,  # Higher default
        threshold: float = 2.0,
        w_decay: float = 0.0001
    ):
        self.lr = learning_rate
        self.threshold = threshold
        self.w_decay = w_decay

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_weights: torch.Tensor,
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:

        phase = kwargs.get("phase", "neutral")
        if phase == "neutral":
            return None, {}

        if local_state is None:
            local_state = {}

        # 1. Activity Estimation (Trace)
        # スパイクがない場合でも、わずかな漏れ（leak）を持たせて勾配消失を防ぎたいが、
        # SNNではスパイクしないと情報がない。
        # そこで、時定数を長くして「過去のスパイク」の影響を長く残す。
        trace_post = local_state.get(
            "trace_post_rate", torch.zeros_like(post_spikes))
        alpha = 0.2  # 0.3 -> 0.2 (Slower decay, more memory)

        activity = trace_post * (1 - alpha) + post_spikes * alpha
        local_state["trace_post_rate"] = activity.detach()

        # 2. Goodness Calculation
        # Goodness = Mean(Activity^2) + small_epsilon
        goodness = activity.pow(2).mean(dim=1) + 1e-6

        # 3. Probability & Factor
        # Sigmoidの勾配が消えないように、threshold近辺に値を寄せる工夫が必要だが、
        # ここは標準的なFFの実装に従う。
        probs = torch.sigmoid(goodness - self.threshold)

        if phase == "positive":
            factor = (1.0 - probs)  # 閾値を超えさせたい
        else:
            factor = (-probs)      # 閾値を下回らせたい

        # 4. Weight Update Calculation
        # delta_w = (activity * factor) @ pre_spikes.T
        # しかし、pre_spikesも疎（sparse）だと更新が起きない。
        # そこで、Pre側もTraceを使うオプションがあるが、計算コストが高い。
        # ここではPost Activityで重み付けする。

        batch_size = pre_spikes.size(0)
        weighted_activity = activity * factor.unsqueeze(1)  # (Batch, N_post)

        # 勾配の方向： 「よく発火したニューロン」への入力を強化/抑制する
        numerator = torch.matmul(weighted_activity.t(), pre_spikes)
        delta_w = numerator / batch_size * self.lr

        # Weight Decay (Regularization)
        delta_w -= self.w_decay * current_weights

        return delta_w, {
            "mean_goodness": goodness.mean().item(),
            "mean_delta_w": delta_w.abs().mean().item()
        }

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ForwardForward",
            "lr": self.lr,
            "threshold": self.threshold
        }
