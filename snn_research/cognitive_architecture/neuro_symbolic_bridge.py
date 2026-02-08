# directory: snn_research/cognitive_architecture
# file: neuro_symbolic_bridge.py
# purpose: 直感(System 1)と論理(System 2)を統合するブリッジモジュール

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

# 修正: SARAAdapter -> SaraAdapter (大文字小文字の修正)
from snn_research.models.adapters.sara_adapter import SaraAdapter

class NeuroSymbolicBridge(nn.Module):
    """
    ニューロシンボリックAIブリッジ:
    SNN（SARAエンジン）による高速な直感・パターン認識（System 1）と、
    シンボリックな推論・論理処理（System 2）を接続するインターフェース。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_dim = config.get("input_dim", 784)
        self.symbolic_dim = config.get("symbolic_dim", 128)
        
        # System 1: SARA Engine (Intuition)
        # 既存のSARAアダプターを利用してRustバックエンドのSNNを駆動
        sara_config = config.get("sara_config", {
            "input_size": self.input_dim,
            "hidden_size": 512,
            "output_size": self.symbolic_dim,
            "enable_rlm": True,
            "enable_attractor": True
        })
        self.system1_snn = SaraAdapter(sara_config)

        # System 2: Symbolic Reasoner (Logic)
        # ここでは単純な線形層としてプレースホルダー実装しているが、
        # 将来的にはグラフニューラルネットワーク(GNN)や論理エンジンへの接続を行う
        self.system2_logic = nn.Sequential(
            nn.Linear(self.symbolic_dim, self.symbolic_dim),
            nn.ReLU(),
            nn.Linear(self.symbolic_dim, 10) # 最終的な判断・分類
        )

        self.threshold = config.get("confidence_threshold", 0.8)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        順伝播:
        1. SNNで直感的な処理を行う
        2. 自信度(Confidence)が高い場合はそのまま出力 (System 1)
        3. 自信度が低い場合は論理層で再考する (System 2)
        """
        # System 1: Fast Intuition
        # SARAエンジンからの出力（スパイク発火率や膜電位）
        intuition_signal = self.system1_snn(x)
        
        # 自信度の計算（最大値やエントロピーなど）
        confidence = torch.max(torch.softmax(intuition_signal, dim=-1), dim=-1)[0]
        
        # System 2: Slow Logic (条件付き実行)
        # バッチ処理のため、実際にはマスク処理や分岐が必要だが、ここでは簡易的に全結合
        logic_output = self.system2_logic(intuition_signal)
        
        # 統合戦略: 重み付け平均 (Gating)
        # confidenceが高いほどSystem 1を重視、低いほどSystem 2を重視
        gate = confidence.unsqueeze(-1)
        # 次元合わせが必要な場合は調整 (ここでは簡易化)
        if intuition_signal.shape[-1] != logic_output.shape[-1]:
             # 次元が違う場合はLogicを正とする（あるいは射影する）
             final_output = logic_output
        else:
             final_output = gate * intuition_signal + (1 - gate) * logic_output

        return {
            "output": final_output,
            "intuition": intuition_signal,
            "logic": logic_output,
            "confidence": confidence
        }

    def consolidate(self):
        """
        睡眠時などに呼び出し。
        論理的結論を直感（SARAの長期記憶）へフィードバックして蒸留する。
        """
        self.system1_snn.consolidate_memory()
        print("NeuroSymbolicBridge: Logic consolidated into Intuition.")