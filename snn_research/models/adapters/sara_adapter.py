# directory: snn_research/models/adapters
# file: sara_adapter.py
# purpose: Adapter for SARA Engine v9.0
# description: SARAEngineをラップし、実験スクリプトが期待する標準的なインターフェース（forward, training_stepなど）を提供します。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from snn_research.models.experimental.sara_engine import SARAEngine, SARAMemory
from snn_research.config.schema import SARAConfig

class SARAAdapter(nn.Module):
    """
    SARA Engine v9.0 へのアダプター。
    状態管理（Memory）を隠蔽し、単純な入出力インターフェースを提供します。
    """
    def __init__(self, config: SARAConfig):
        super().__init__()
        self.engine = SARAEngine(config)
        self.memory: Optional[SARAMemory] = None

    def reset_state(self):
        """内部メモリと状態をリセット"""
        self.memory = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        推論用フォワードパス。
        行動（Action）または出力のみを返します。
        """
        # SARA v9.0 returns: (action, memory, info)
        action, self.memory, _ = self.engine(x, prev_memory=self.memory)
        return action

    def training_step(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        学習ステップを実行し、メトリクスを返します。
        """
        # SARA v9.0 adapt returns dict with keys: loss, surprise, memory, outputs, info
        results = self.engine.adapt(x, targets=y, prev_memory=self.memory)
        
        # 内部メモリを更新
        self.memory = results['memory']
        
        return {
            "loss": results['loss'],
            "output": results['outputs'],
            "surprise": results['surprise'],
            "energy": results['info'].get('energy', 0.0),
            "free_energy": results['info'].get('free_energy', 0.0)
        }

# Backward compatibility alias
SaraAdapter = SARAAdapter