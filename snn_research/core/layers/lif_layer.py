# ファイルパス: snn_research/core/layers/lif_layer.py
# 日本語タイトル: Standard LIF Layer (Fixed)
# 目的・内容:
#   行列演算を用いた標準的なLIFレイヤーの実装。
#   ループ処理による形状不一致エラーを解消。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
from torch import Tensor

from snn_research.core.layers.abstract_snn_layer import AbstractSNNLayer


class LIFLayer(AbstractSNNLayer):
    """
    標準的なLIFニューロン層。
    入力に対して線形変換(重み乗算)を行い、LIFニューロンに入力する。
    """

    def __init__(self, input_features: int = 784, neurons: int = 100, name: str = "lif", **kwargs):
        super().__init__(name=name)
        self.input_features = input_features
        self._neurons = neurons # AbstractSNNLayerのneuronsプロパティと整合
        self.kwargs = kwargs
        
        # パラメータ定義
        # nn.Linearを使用せず、重みを直接管理する場合
        self.W = nn.Parameter(torch.randn(neurons, input_features) * 0.01)
        self.b = nn.Parameter(torch.zeros(neurons))
        
        # 状態変数
        self.v = None
        self.register_buffer('membrane_potential', torch.zeros(1, neurons))

        # ハイパーパラメータ
        self.decay = kwargs.get("decay", 0.9)
        self.threshold = kwargs.get("threshold", 1.0)
        
        self.built = True

    @property
    def neurons(self) -> int:
        return self._neurons

    def build(self) -> None:
        self.built = True

    def reset_state(self) -> None:
        if self.membrane_potential is not None:
            self.membrane_potential.fill_(0.0)

    def forward(self, inputs: Tensor, model_state: Optional[Dict[str, Tensor]] = None) -> Dict[str, Tensor]:
        """
        順伝播処理。
        行列演算(F.linear)を使用してシナプス入力を計算する。
        """
        if not self.built:
            self.build()

        batch_size = inputs.shape[0]
        
        # 状態の初期化確認
        if self.membrane_potential.shape[0] != batch_size:
            self.membrane_potential = torch.zeros(
                batch_size, self._neurons, device=inputs.device)

        # 1. シナプス入力 (Synaptic Input)
        # I = W * x + b
        # inputs: [Batch, InFeatures]
        # W: [OutFeatures, InFeatures]
        synaptic_input = F.linear(inputs, self.W, self.b)

        # 2. 膜電位更新 (Leaky Integration)
        # v[t] = v[t-1] * decay + I[t]
        self.membrane_potential = self.membrane_potential * self.decay + synaptic_input

        # 3. 発火 (Spike Generation)
        spikes = (self.membrane_potential >= self.threshold).float()

        # 4. リセット (Reset)
        # Soft Reset: v = v - threshold
        self.membrane_potential = self.membrane_potential - (spikes * self.threshold)

        return {
            "spikes": spikes,
            "activity": spikes, # 互換性のため
            "membrane_potential": self.membrane_potential
        }