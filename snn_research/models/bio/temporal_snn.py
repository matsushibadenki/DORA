# directory: snn_research/models/bio
# filename: temporal_snn.py
# description: 時系列データ処理SNN (後方互換性エイリアス追加版)

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

# LIFニューロン定義 (前回と同様)
try:
    from snn_research.core.neurons.lif_neuron import LIFNeuron
except ImportError:
    class LIFNeuron(nn.Module):
        def __init__(self, tau=2.0):
            super().__init__()
            self.tau = tau
            self.threshold = 1.0
        def forward(self, x):
            return (x > self.threshold).float()

class TemporalSNN(nn.Module):
    """
    TemporalSNN:
    時系列入力（音声、心電図、時系列センサーデータなど）を処理するための
    再帰結合を持つスパイキングニューラルネットワーク。
    (旧名: SimpleRSNN)
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        super().__init__()
        self.config = config if isinstance(config, dict) else kwargs
        
        self.input_size = self.config.get("input_size", 128)
        self.hidden_size = self.config.get("hidden_size", 256)
        self.output_size = self.config.get("output_size", 10)
        self.time_steps = self.config.get("time_steps", 16)
        
        # 入力層 -> 隠れ層
        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.lif_in = LIFNeuron()
        
        # 再帰結合 (Recurrent)
        self.fc_rec = nn.Linear(self.hidden_size, self.hidden_size)
        self.lif_rec = LIFNeuron()
        
        # 出力層
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        
        nn.init.kaiming_normal_(self.fc_in.weight)
        nn.init.orthogonal_(self.fc_rec.weight)
        nn.init.xavier_normal_(self.fc_out.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理
        Args:
            x: [Batch, Time, Features] or [Batch, Features]
        Returns:
            out: [Batch, Output_Size]
        """
        batch_size = x.shape[0]
        
        # 入力が [Batch, Features] の場合は時間次元を追加
        if x.dim() == 2:
            x = x.unsqueeze(1).repeat(1, self.time_steps, 1)
            
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        spike_record = []
        
        # 時間ステップごとの処理
        steps = x.shape[1] if x.dim() > 1 else self.time_steps
        for t in range(steps):
            input_t = x[:, t, :] if x.dim() > 2 else x
            
            current = self.fc_in(input_t) + self.fc_rec(h)
            spike = self.lif_rec(current)
            h = spike
            spike_record.append(spike)
            
        spike_stack = torch.stack(spike_record, dim=1)
        mean_firing_rate = spike_stack.mean(dim=1)
        out = self.fc_out(mean_firing_rate)
        
        return out

# --- 後方互換性のためのエイリアス ---
# これにより `from .temporal_snn import SimpleRSNN` が成功するようになります
SimpleRSNN = TemporalSNN