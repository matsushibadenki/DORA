# ファイルパス: snn_research/learning_rules/base_rule.py
# 日本語タイトル: Base Plasticity Rule Interface (Fixed)
# 目的・内容:
#   互換性エイリアス "BioLearningRule" を追加。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
from torch import Tensor


class PlasticityRule(nn.Module, ABC):
    """
    全ての学習則の基底クラス。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.config = kwargs

    @abstractmethod
    def update(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        current_weights: Tensor,
        **kwargs: Any
    ) -> Tuple[Optional[Tensor], Dict[str, Any]]:
        pass

    def reset_state(self) -> None:
        pass


# --- Backward Compatibility ---
BioLearningRule = PlasticityRule