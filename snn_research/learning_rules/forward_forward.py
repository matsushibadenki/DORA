# ファイルパス: snn_research/learning_rules/forward_forward.py
# 日本語タイトル: Forward-Forward Learning Rule Module
# 目的・内容:
#   HintonのForward-Forward AlgorithmをSNN向けに適合させた実装。
#   グローバルな誤差逆伝播を使わず、局所的な「Positive/Negative」フェーズ情報のみで重みを更新する。

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from snn_research.learning_rules.base_rule import PlasticityRule


class ForwardForwardRule(PlasticityRule):
    """
    Forward-Forward Learning Rule for SNN.
    
    設計方針書 4.1準拠:
    - 局所学習のみを使用
    - Phase ("positive", "negative") に基づく重み更新
    """

    def __init__(self, learning_rate: float = 0.01, threshold: float = 2.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lr = learning_rate
        self.threshold = threshold

    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        Args:
            phase (str): "positive" (wake/reinforced) or "negative" (sleep/noise)
        """
        phase = kwargs.get("phase", "neutral")
        
        # 学習対象外のフェーズでは更新しない
        if phase == "neutral":
            return None, {}

        # SNNにおけるGoodnessの定義:
        # Pre発火とPost発火の同時生起（Heavyside的な活動度）をGoodnessの代理とする
        # batch次元(0)で平均を取らず、各サンプルの寄与を計算
        
        # shape: (out_features, in_features)
        # pre: (batch, in), post: (batch, out) -> outer product -> (batch, out, in)
        # ここでは簡易的に einsum でバッチごとの相関を計算
        
        # 活動が高い = Goodnessが高い
        # Positive Phase: 活動を上げる方向へ (Hebbian)
        # Negative Phase: 活動を下げる方向へ (Anti-Hebbian)
        
        # 重み更新量 ΔW
        # ΔW = direction * lr * (post^T @ pre)
        
        activity_product = torch.matmul(post_spikes.T, pre_spikes)
        # batchサイズで正規化
        batch_size = pre_spikes.shape[0]
        if batch_size > 0:
            activity_product = activity_product / batch_size

        if phase == "positive":
            # 正のフェーズ: 相関を強める（閾値を超えさせる）
            delta_w = self.lr * activity_product
        elif phase == "negative":
            # 負のフェーズ: 相関を弱める（閾値を下回らせる）
            # ここでは単純なAnti-Hebbianとして実装
            delta_w = -self.lr * activity_product
        else:
            return None, {}

        logs = {
            "ff_phase": phase,
            "mean_delta": delta_w.abs().mean().item()
        }
        
        return delta_w, logs