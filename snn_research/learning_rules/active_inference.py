# ファイルパス: snn_research/learning_rules/active_inference.py
# 日本語タイトル: Active Inference Learning Rule Module
# 目的・内容:
#   能動的推論（自由エネルギー原理）に基づく局所学習則。
#   トップダウン予測とボトムアップ入力の誤差（予測誤差）を最小化する方向へ重みを更新する。

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor
from snn_research.learning_rules.base_rule import PlasticityRule


class ActiveInferenceRule(PlasticityRule):
    """
    Active Inference / Predictive Coding Rule.
    
    設計方針書 4.1準拠:
    - 予測誤差最小化
    - 推論 = 状態遷移
    """

    def __init__(self, learning_rate: float = 0.005, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.lr = learning_rate

    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        local_state: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        """
        このルールでは、Postニューロンが「予測（Prediction）」、
        Preニューロン入力×重みが「観測（Observation）」またはその逆の関係にあると仮定する文脈が多いが、
        ここではSNNの文脈で「Postニューロンへの入力電流（予測）」と「Postニューロンの実際の活動（観測）」
        の不整合を減らす形、あるいはトップダウン信号との誤差を扱う。
        
        簡易実装として:
        Feedback信号(target_signal)がある場合、それと実際のPost発火の誤差を最小化する。
        """
        target_signal = kwargs.get("target_signal", None)
        
        # 教師信号やフィードバックがない場合は、自身の発火活動を維持するホメオスタシス的挙動、
        # あるいは何もしない
        if target_signal is None:
            return None, {}

        # Postスパイクとターゲット（期待される活動）の差分
        # target_signalは発火率またはスパイクそのもの
        prediction_error = target_signal - post_spikes

        # 誤差に基づき重みを更新
        # Preが発火していて、かつ誤差がある場合に重みを修正
        # ΔW = lr * error * pre
        
        # (batch, out) -> (batch, out, 1)
        # (batch, in)  -> (batch, 1, in)
        # matmul -> (batch, out, in)
        delta_w = torch.matmul(prediction_error.unsqueeze(2), pre_spikes.unsqueeze(1))
        
        delta_w = delta_w.mean(dim=0) * self.lr

        return delta_w, {"pred_error": prediction_error.abs().mean().item()}