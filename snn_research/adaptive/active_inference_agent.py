# directory: snn_research/adaptive
# file: active_inference_agent.py
# purpose: Active Inference Agent implementation
# description: 能動的推論エージェント。
#              SARA Engine v9.0 への統合に伴い、内部ロジックを SARA の ActiveInferenceController を利用するように変更、
#              または独立したラッパーとして整理します。

import torch
import torch.nn as nn
from typing import Optional

class ActiveInferenceAgent:
    """
    Wrapper around a model to perform Active Inference.
    """
    def __init__(self, model: nn.Module, state_dim: int, action_dim: int):
        self.model = model
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 行動生成用の追加レイヤー（モデルが持っていない場合）
        if not hasattr(self.model, 'infer_action'):
            self.action_head = nn.Linear(state_dim, action_dim)
        else:
            self.action_head = None

    def infer_action(self, observation: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        観測から行動を推論する
        """
        # モデルの推論（状態推定）
        # モデルの出力形式に応じて処理を分岐
        if hasattr(self.model, 'forward'):
            output = self.model(observation)
            # outputがタプルの場合（SARAなど）、状態部分を抽出するロジックが必要だが
            # ここでは簡易的に output そのものを状態とみなすか、モデルが適切に処理すると仮定
            if isinstance(output, tuple):
                state = output[0]
            else:
                state = output
        else:
             # Fallback
             state = observation

        # 行動生成
        if self.action_head:
            action = torch.tanh(self.action_head(state))
        elif hasattr(self.model, 'infer_action'):
            action = self.model.infer_action(state, goal)
        else:
            # Default random action if no mechanism
            action = torch.randn(observation.size(0), self.action_dim)
            
        return action