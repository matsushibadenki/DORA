# ファイルパス: snn_research/io/spike_decoder.py
# 日本語タイトル: Spike Decoder Implementation (Fixed)
# 目的・内容:
#   - SpikeDecoderの初期化引数 output_dim を任意(Optional)に変更。
#   - RateDecoder等が次元指定なしでインスタンス化できるようにする。

import torch
import torch.nn as nn
from typing import Optional, List, Union

class SpikeDecoder(nn.Module):
    """
    スパイク列をデコードして値に戻すための基底クラス。
    """
    # 修正: output_dim を Optional に変更
    def __init__(self, output_dim: Optional[int] = None, device: str = 'cpu'):
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
    次元は維持されるため、output_dim = input_features となる。
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
    TTFS (Time-to-First-Spike) デコーディング。
    """
    def __init__(self, output_dim: int, duration: int, device: str = 'cpu'):
        super().__init__(output_dim, device)
        self.duration = duration

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        if spikes.dim() != 3:
            return spikes 

        times = torch.arange(spikes.size(1), device=self.device).view(1, -1, 1)
        spike_times = torch.where(spikes > 0.5, times, torch.tensor(float('inf'), device=self.device))
        first_spike_time, _ = spike_times.min(dim=1)
        
        decoded = (self.duration - first_spike_time).clamp(min=0) / self.duration
        return decoded

class LinearReadoutDecoder(SpikeDecoder):
    """
    学習可能な線形読み出し層。
    """
    def __init__(self, input_features: int, output_dim: int, device: str = 'cpu'):
        super().__init__(output_dim, device)
        self.readout = nn.Linear(input_features, output_dim).to(device)
    
    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        rate = spikes.mean(dim=1) if spikes.dim() == 3 else spikes
        return self.readout(rate)