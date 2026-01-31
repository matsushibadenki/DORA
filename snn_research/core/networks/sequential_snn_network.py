# ファイルパス: snn_research/core/networks/sequential_snn_network.py
# 日本語タイトル: Sequential SNN Network (Missing Import Fixed)
# 目的・内容:
#   Sequentialモデルの実装。
#   mypyエラー "Name 'Tensor' is not defined" を修正。

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Dict, Any, Optional, cast, Union, OrderedDict

from snn_research.core.network import AbstractNetwork
from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer


class SequentialSNN(AbstractNetwork):
    """
    複数のSNNレイヤーを直列に接続したネットワークモデル。
    """

    def __init__(
        self, 
        layers: Optional[Union[List[AbstractSNNLayer], OrderedDict[str, nn.Module]]] = None
    ) -> None:
        super().__init__()
        
        if layers:
            if isinstance(layers, list):
                self.layers.extend(layers)
            elif isinstance(layers, OrderedDict):
                # OrderedDictの場合は値（レイヤー）をリストとして登録
                self.layers.extend(list(layers.values()))
            else:
                pass
                
        self.built = True

    def add(self, layer: AbstractSNNLayer) -> None:
        """レイヤーを追加する"""
        self.layers.append(layer)

    def forward(self, x: Tensor) -> Tensor:
        """
        順伝播計算。
        """
        current_input = x
        model_state: Dict[str, Any] = {}

        for layer in self.layers:
            if isinstance(layer, AbstractSNNLayer):
                out = layer(current_input, model_state)
                
                if isinstance(out, dict):
                    if 'activity' in out:
                        current_input = out['activity']
                    elif 'spikes' in out:
                        current_input = out['spikes']
                    elif 'output' in out:
                        current_input = out['output']
                    else:
                        current_input = list(out.values())[0]
                else:
                    current_input = out
            else:
                # 通常の nn.Module
                current_input = layer(current_input)

        return current_input

    def reset_state(self) -> None:
        """全レイヤーの状態をリセット"""
        for layer in self.layers:
            l = cast(Any, layer)
            if hasattr(l, 'reset_state'):
                l.reset_state()
            elif hasattr(l, 'reset'):
                l.reset()

    def get_layer_states(self) -> List[Optional[torch.Tensor]]:
        """各レイヤーの内部状態を取得"""
        states = []
        for layer in self.layers:
            if hasattr(layer, 'membrane_potential'):
                states.append(getattr(layer, 'membrane_potential'))
            else:
                states.append(None)
        return states


# --- Backward Compatibility Alias ---
SequentialSNNNetwork = SequentialSNN