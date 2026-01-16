# ファイルパス: snn_research/cognitive_architecture/som_feature_map.py
# 修正: STDPRule対応、update戻り値処理の修正

import torch
import torch.nn as nn
from typing import Tuple

# 新しいクラス名をインポート (エイリアスSTDPも使えるが、明示的に新しい方を使う)
from snn_research.learning_rules.stdp import STDPRule

class SomFeatureMap(nn.Module):
    """
    STDPを用いて特徴を自己組織化する、単層のSNN。
    """
    def __init__(self, input_dim: int, map_size: Tuple[int, int], stdp_params: dict):
        super().__init__()
        self.input_dim = input_dim
        self.map_size = map_size
        self.num_neurons = map_size[0] * map_size[1]
        
        self.weights = nn.Parameter(torch.rand(self.input_dim, self.num_neurons))
        
        # STDPRuleを使用。kwargsで余計なパラメータは吸収される。
        self.stdp = STDPRule(**stdp_params)
        
        self.neuron_pos = torch.stack(torch.meshgrid(
            torch.arange(map_size[0]),
            torch.arange(map_size[1]),
            indexing='xy'
        )).float().reshape(2, -1).T
        
        print(f"🗺️ 自己組織化マップが初期化されました ({map_size[0]}x{map_size[1]})。")

    def forward(self, input_spikes: torch.Tensor) -> torch.Tensor:
        # デバイス同期
        if input_spikes.device != self.weights.device:
            input_spikes = input_spikes.to(self.weights.device)

        activation = input_spikes @ self.weights
        winner_index = torch.argmax(activation)
        
        output_spikes = torch.zeros(self.num_neurons, device=input_spikes.device)
        output_spikes[winner_index] = 1.0
        
        return output_spikes

    def update_weights(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor):
        """
        STDPと近傍学習則に基づき、重みを更新する。
        """
        if pre_spikes.device != self.weights.device:
            pre_spikes = pre_spikes.to(self.weights.device)
        if post_spikes.device != self.weights.device:
            post_spikes = post_spikes.to(self.weights.device)

        winner_index = torch.argmax(post_spikes)
        
        # 1. 近傍関数
        if self.neuron_pos.device != self.weights.device:
            self.neuron_pos = self.neuron_pos.to(self.weights.device)
            
        distances = torch.linalg.norm(self.neuron_pos - self.neuron_pos[winner_index], dim=1)
        neighborhood_factor = torch.exp(-distances**2 / (2 * (self.map_size[0]/4)**2))
        
        # 2. STDPベースの重み更新
        # STDPRule.update は (delta_w, logs) を返す
        # self.weights.T を渡しているため、返り値 dw_transposed は (N_post, N_pre)
        result = self.stdp.update(pre_spikes, post_spikes, self.weights.T)
        
        # 結果の検証
        dw_transposed, _ = result
        
        if dw_transposed is None:
            return

        dw = dw_transposed.T # (N_pre, N_post)
        
        # 3. 近傍関数で学習率を変調
        # dw: (N_in, N_out), neighborhood_factor: (N_out)
        modulated_dw = dw * neighborhood_factor
        
        self.weights.data += modulated_dw
        self.weights.data = torch.clamp(self.weights.data, 0, 1)