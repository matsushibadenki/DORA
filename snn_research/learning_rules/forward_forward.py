# ファイルパス: snn_research/learning_rules/forward_forward.py
# 日本語タイトル: Forward-Forward Learning Rule
# 目的・内容:
#   Hinton (2022) のForward-ForwardアルゴリズムのSNN向け実装。
#   誤差逆伝播を使わず、局所的な「Goodness（良さ）」関数の勾配に従って学習する。
#   Positiveフェーズ（正解データ）とNegativeフェーズ（睡眠/偽データ）を切り替えて使用する。

import torch
import logging
from typing import Dict, Any, Tuple, Optional

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)

class ForwardForwardRule(PlasticityRule):
    """
    Forward-Forward Learning Rule.
    
    各層は独立して以下の目的関数を最適化する:
    Pos Phase: maximize goodness
    Neg Phase: minimize goodness
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
        FF学習則による更新。
        kwargsに 'phase': 'positive' | 'negative' が必要。
        """
        phase = kwargs.get("phase", "neutral")
        if phase == "neutral":
            return None, {}

        # SNNにおける「Activity」の定義:
        # スパイクそのもの、または膜電位、あるいは短時間レートを使用する。
        # ここではスパイク列の移動平均（レート）をActivityとして近似する。
        
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
        # Pos: Goodness > Threshold にしたい
        # Neg: Goodness < Threshold にしたい
        # Loss関数として記述すると:
        # L_pos = log(1 + exp(-(goodness - threshold)))
        # L_neg = log(1 + exp(+(goodness - threshold)))
        
        # 簡易的な実装:
        # delta_activity ~ sign * (goodness - threshold)
        
        sign = 1.0 if phase == "positive" else -1.0
        
        # 重み更新の方向:
        # dG/dw = dG/da * da/du * du/dw
        # u = W * x (入力電流)
        # da/du は活性化関数の微分だが、SNNでは近似的に 1 または surrogate を使う。
        # ここでは単純に Hebbian term (post * pre) に Goodness 誤差を乗じる。
        
        # Error signal: (Batch, N_post)
        # 各ニューロンがGoodnessにどれだけ寄与したか * 全体の誤差
        # (Batch, 1) * (Batch, N_post)
        
        # thresholdとの差分を誤差信号とする
        error_signal = (goodness - self.threshold).unsqueeze(1) # (Batch, 1)
        
        # Positive: error < 0 (Goodnessが足りない) なら増やしたい -> sign=+1, error負 -> 全体負??
        # 整理:
        # maximize J = sum( (G - theta) ) for pos
        # minimize J = sum( (G - theta) ) for neg
        # grad = (G - theta) * activity * pre
        
        # 正しくは
        # delta W = lr * sign * (goodness - threshold) * activity * pre
        # しかしこれだと、Goodnessが高いとさらに高めようとしてしまう。
        # Hintonの論文のSigmoid cross entropyの微分に近い形を採用する。
        
        probs = torch.sigmoid(goodness - self.threshold)
        # pos: 1に近づけたい -> grad ~ (1 - probs)
        # neg: 0に近づけたい -> grad ~ (0 - probs) = -probs
        
        factor = (1.0 - probs) if phase == "positive" else (-probs)
        factor = factor.unsqueeze(1).unsqueeze(2) # (Batch, 1, 1)
        
        # (Batch, N_post, 1) * (Batch, 1, N_pre) -> (Batch, N_post, N_pre)
        # activity: (Batch, N_post)
        # pre_spikes: (Batch, N_pre)
        
        eligibility = torch.einsum("bi,bj->bij", activity, pre_spikes)
        
        delta_w_batch = factor * eligibility
        delta_w = delta_w_batch.mean(dim=0) * self.lr

        # Weight Decay
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