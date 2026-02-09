# directory: snn_research/training/trainers
# file: self_supervised.py
# title: Self-Supervised Trainer (SARA Integrated)
# purpose: ラベルなしデータから環境の内的モデルを構築するトレーナー。
#          SARA Engineの予測符号化機能を用いて、データの内的表現を自己学習する。

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any

try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    SARAEngine = None

class SelfSupervisedTrainer:
    """SARA Engine のための自己教師あり学習トレーナー"""
    def __init__(self, model: nn.Module, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.is_sara = SARAEngine and isinstance(model, SARAEngine)
        if self.is_sara:
            # SARAを予測符号化モードに固定
            self.model.set_perception_mode("predictive_coding")
            print("SelfSupervisedTrainer: SARA Engine Backend Enabled (Predictive Mode)")

    def train_step(self, data: torch.Tensor) -> Dict[str, float]:
        """1ステップの学習を実行"""
        self.model.train()
        data = data.to(self.device)
        self.optimizer.zero_grad()
        
        if self.is_sara:
            # SARA内部の process_step により予測誤差（自由エネルギー）を取得
            results = self.model.process_step(
                input_data=data,
                require_prediction=True
            )
            # 内部損失（自由エネルギー等）を最小化
            loss = results.get("loss") or results.get("free_energy")
        else:
            # Legacy: AutoEncoder
            recon = self.model(data)
            loss = nn.functional.mse_loss(recon, data)
            
        if loss is not None and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            return {"loss": loss.item()}
        
        return {"loss": 0.0}