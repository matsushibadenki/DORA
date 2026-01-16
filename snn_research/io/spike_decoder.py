# ファイルパス: snn_research/io/spike_decoder.py
# 日本語タイトル: スパイクデコーダ (Implementation)
# 目的・内容:
#   - スパイク列を連続値やクラスラベルに復元するためのデコーダ群を定義。
#   - RateDecoder, LinearReadoutDecoderなどを実装。

import torch
import torch.nn as nn
from typing import Optional, List, Union

class SpikeDecoder(nn.Module):
    """
    スパイク列をデコードして値に戻すための基底クラス。
    """
    def __init__(self, output_dim: int, device: str = 'cpu'):
        super().__init__()
        self.output_dim = output_dim
        self.device = device

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spikes (Tensor): (Batch, Time, Features) または (Batch, Features)
        Returns:
            decoded (Tensor): (Batch, OutputDim)
        """
        raise NotImplementedError

class RateDecoder(SpikeDecoder):
    """
    レートデコーディング: 時間方向の平均発火率を出力とする。
    """
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (Batch, Time, Features)
        if spikes.dim() == 3:
            # 時間平均を取る
            return spikes.mean(dim=1)
        elif spikes.dim() == 2:
            # 時間次元がない場合はそのまま（瞬時レートとみなす）
            return spikes
        else:
            raise ValueError(f"Unexpected spike shape: {spikes.shape}")

class FirstSpikeDecoder(SpikeDecoder):
    """
    TTFS (Time-to-First-Spike) デコーディング: 最初に発火した時刻を値とする。
    （早い発火 = 大きな値、発火なし = 0）
    """
    def __init__(self, output_dim: int, duration: int, device: str = 'cpu'):
        super().__init__(output_dim, device)
        self.duration = duration

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # spikes: (Batch, Time, Features)
        if spikes.dim() != 3:
            return spikes # フォールバック

        # 最初の発火時刻を探す
        # 発火していない場所は duration とする
        times = torch.arange(spikes.size(1), device=self.device).view(1, -1, 1)
        
        # 発火した瞬間の時刻を取得 (発火なければ無限大扱い)
        spike_times = torch.where(spikes > 0.5, times, torch.tensor(float('inf'), device=self.device))
        
        # 最小時刻を取得 (Batch, Features)
        first_spike_time, _ = spike_times.min(dim=1)
        
        # 値に変換: (Duration - Time) / Duration => 0~1
        decoded = (self.duration - first_spike_time).clamp(min=0) / self.duration
        return decoded

class LinearReadoutDecoder(SpikeDecoder):
    """
    学習可能な線形読み出し層。
    スパイクのレートを入力とし、線形変換を行う。
    """
    def __init__(self, input_features: int, output_dim: int, device: str = 'cpu'):
        super().__init__(output_dim, device)
        self.readout = nn.Linear(input_features, output_dim).to(device)
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        # レート計算
        rate = spikes.mean(dim=1) if spikes.dim() == 3 else spikes
        return self.readout(rate)