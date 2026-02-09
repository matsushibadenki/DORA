# directory: snn_research/adaptive
# file: test_time_adaptation.py
# purpose: Test-time adaptation logic
# description: 推論時（テスト時）に適応を行うためのラッパーモジュール。
#              Pytestの警告を回避するためクラス名を変更 (TestTime... -> Time...)

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional, Callable

class TimeAdaptationWrapper(nn.Module):
    """
    Wraps a model to enable adaptation during test time (inference).
    """
    def __init__(self, model: nn.Module, learning_rate: float = 1e-4):
        super().__init__()
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def adapt(self, x: torch.Tensor, loss_fn: Callable[[torch.Tensor], torch.Tensor]):
        """
        Update model parameters based on a self-supervised loss function.
        """
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = loss_fn(output)
        loss.backward()
        self.optimizer.step()
        return loss