# ファイルパス: snn_research/learning_rules/forward_forward.py
# 日本語タイトル: Forward-Forward Learning Rule Implementation
# 目的・内容:
#   Neuromorphic OSにおける主要学習則の一つ。
#   Hinton (2022) のForward-ForwardアルゴリズムをSNN向けに実装。
#   Positive/Negativeフェーズによる局所的なGoodness最適化を行う。

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class ForwardForwardRule(PlasticityRule):
    """
    Forward-Forward Learning Rule.
    
    ニューラルネットワークの各層が独立して「Goodness」を最大化・最小化するように学習する。
    誤差逆伝播（Backprop）を使用しないため、生物学的妥当性が高く、並列化に適している。
    
    Phases:
      - Positive Phase (wake): maximize goodness (正解データ)
      - Negative Phase (sleep/dream): minimize goodness (生成データ/ノイズ)
      
    Goodness = sum(activity^2)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        threshold: float = 2.0,  # Goodnessの目標閾値
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
        """
        FF学習則による重み更新の計算。
        
        Args:
            pre_spikes: 前シナプスニューロンのスパイク (Batch, N_pre)
            post_spikes: 後シナプスニューロンのスパイク (Batch, N_post)
            current_weights: 現在の重み行列
            local_state: シナプス局所状態（トレースなど）
            kwargs: 'phase' ('positive' or 'negative') を含む必要がある
            
        Returns:
            delta_w: 重み更新量
            logs: ログ情報
        """
        phase = kwargs.get("phase", "neutral")
        if phase == "neutral":
            return None, {}

        # SNNにおける「Activity」の定義:
        # スパイクそのものだと疎すぎるため、移動平均（レート）をActivityとして近似する。
        
        if local_state is None:
            local_state = {}
            
        # Activity (Rate) 推定
        # trace_post: (Batch, N_post)
        trace_post = local_state.get("trace_post_rate", torch.zeros_like(post_spikes))
        alpha = 0.3 # 平滑化係数
        
        # レート更新
        activity = trace_post * (1 - alpha) + post_spikes * alpha
        local_state["trace_post_rate"] = activity.detach()

        # Goodness 計算: G = sum(activity^2) per sample
        goodness = activity.pow(2).mean(dim=1) # (Batch,)
        
        # 勾配の方向決定
        # Positive Phase: Goodness > Threshold にしたい
        # Negative Phase: Goodness < Threshold にしたい
        
        # 確率的勾配の近似
        # probs = sigmoid(Goodness - threshold)
        probs = torch.sigmoid(goodness - self.threshold)
        
        # 重み更新係数 (Hintonの論文に基づく符号設計)
        # pos: 1に近づけたい -> factor ~ (1 - probs)
        # neg: 0に近づけたい -> factor ~ (0 - probs) = -probs
        factor = (1.0 - probs) if phase == "positive" else (-probs)
        factor = factor.unsqueeze(1).unsqueeze(2) # (Batch, 1, 1)
        
        # Hebbian Term: activity * pre_input
        # (Batch, N_post, 1) * (Batch, 1, N_pre) -> (Batch, N_post, N_pre)
        eligibility = torch.einsum("bi,bj->bij", activity, pre_spikes)
        
        # 更新量の計算
        delta_w_batch = factor * eligibility
        delta_w = delta_w_batch.mean(dim=0) * self.lr

        # Weight Decay (忘却)
        delta_w -= self.w_decay * current_weights

        logs = {
            "mean_goodness": goodness.mean().item(),
            "mean_prob": probs.mean().item(),
            "phase": phase
        }

        return delta_w, logs

    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "ForwardForward",
            "lr": self.lr,
            "threshold": self.threshold
        }