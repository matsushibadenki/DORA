# ファイルパス: snn_research/core/network.py
# 日本語タイトル: Abstract Network Interface (Flexible Return Type)
# 目的・内容:
#   SNNモデルや認知アーキテクチャの統一インターフェース。
#   forwardメソッドの戻り値を柔軟化。

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, cast, Union

import torch
import torch.nn as nn
from torch import Tensor

# 絶対パスインポート
from snn_research.layers.abstract_layer import AbstractLayer


class AbstractNetwork(nn.Module, ABC):
    """
    全てのニューラルネットワークモデルの基底クラス。
    """
    
    def __init__(self, layers: Optional[List[AbstractLayer]] = None) -> None:
        super().__init__()
        # ModuleListに登録することでパラメータ管理を自動化
        self.layers = nn.ModuleList(layers or [])
        self.built: bool = False

    @abstractmethod
    def forward(self, x: Tensor) -> Union[Tensor, Dict[str, Tensor]]:
        """
        順伝播の定義。
        単一のTensor（予測結果など）または 辞書（状態や複数の出力）を返すことができる。
        """
        pass

    def update_model(
        self,
        inputs: Tensor,
        targets: Optional[Tensor],
        model_state: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        """
        モデルのパラメータ更新を行うインターフェース。
        """
        all_metrics: Dict[str, Tensor] = {}
        
        # デフォルトの実装例
        if self.training:
            pass

        return all_metrics

    def reset_state(self) -> None:
        """モデルの状態をリセットする"""
        for layer in self.layers:
            if hasattr(layer, 'reset_state'):
                cast(Any, layer).reset_state()