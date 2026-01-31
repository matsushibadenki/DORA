# snn_research/learning_rules/probabilistic_hebbian.py
# 修正: 戻り値の型アノテーションを基底クラス(PlasticityRule)と一致させる。

from __future__ import annotations

import torch
from typing import Dict, Any, Optional, Tuple
from .base_rule import BioLearningRule


class ProbabilisticHebbian(BioLearningRule):
    """
    確率的ヘブ則の実装。
    """
    
    def __init__(self, learning_rate: float = 0.005, weight_decay: float = 0.0001):
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        current_weights: torch.Tensor,
        **kwargs: Any
    ) -> Tuple[Optional[torch.Tensor], Dict[str, Any]]:
        """
        基底クラスに合わせて戻り値は (Optional[Tensor], Dict[str, Any]) とする。
        """

        # バッチ次元の処理
        if pre_spikes.dim() == 1:
            pre_spikes = pre_spikes.unsqueeze(0)
        if post_spikes.dim() == 1:
            post_spikes = post_spikes.unsqueeze(0)

        # ヘブ則: Δw = η * (post * pre - decay * w)
        hebbian_term = torch.bmm(
            post_spikes.unsqueeze(2), pre_spikes.unsqueeze(1))
        
        mean_hebbian = hebbian_term.mean(dim=0)
        decay_term = self.weight_decay * current_weights

        dw = self.learning_rate * (mean_hebbian - decay_term)
        
        # 第2戻り値は辞書型を返す必要がある
        return dw, {}