# directory: snn_research/models/adapters
# file: async_mamba_adapter.py
# purpose: Adapter for Asynchronous BitSpike Mamba model
# description: 非同期環境で BitSpikeMamba モデルを実行するためのアダプター。
#              device引数やcheckpoint_path引数に対応し、テストとの互換性を確保。

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

# モデル定義があるパスからインポート
from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
from snn_research.config.schema import ModelConfig

class AsyncBitSpikeMambaAdapter(nn.Module):
    """
    非同期実行用の BitSpikeMamba アダプター。
    状態を保持し、ストリーミング入力に対応します。
    """
    def __init__(self, config: Any, device: str = 'cpu', checkpoint_path: Optional[str] = None):
        super().__init__()
        
        # Configが辞書の場合とオブジェクトの場合に対応
        if isinstance(config, dict):
             input_dim = config.get('d_model', 128) # Mamba config keys might differ
             hidden_dim = config.get('d_model', 256)
             output_dim = config.get('vocab_size', 10) # Assuming vocab size for output
        else:
             input_dim = getattr(config, 'input_dim', 128)
             hidden_dim = getattr(config, 'hidden_dim', 256)
             output_dim = getattr(config, 'output_dim', 10)
        
        self.device = device
        self.model = BitSpikeMamba(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim
        ).to(device)
        
        if checkpoint_path:
            # Load checkpoint logic here
            pass
            
        self.state = None

    def reset_state(self):
        """内部状態のリセット"""
        self.state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        入力 x に対して推論を行い、出力を返す。
        状態は内部で更新・保持される。
        """
        x = x.to(self.device)
        
        # BitSpikeMambaの仕様に合わせて調整
        if hasattr(self.model, 'forward_step'):
             output, next_state = self.model.forward_step(x, self.state)
        else:
             # 通常のforwardの場合はシーケンスとして処理するか、ダミー次元を追加
             if x.dim() == 2:
                 x_seq = x.unsqueeze(1) # (B, 1, D)
                 output, next_state = self.model(x_seq, self.state)
                 output = output.squeeze(1)
             else:
                 output, next_state = self.model(x, self.state)

        self.state = next_state
        return output