# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace Module v2.1 (Device Safe)
# 目的・内容:
#   デバイス不一致（MPS vs CPU）によるランタイムエラーを防ぐための安全装置を追加。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory.
    """
    def __init__(self, dim: int = 64, decay: float = 0.9):
        super().__init__()
        self.dim = dim
        self.decay = decay
        
        # Current conscious content (The "Thought")
        # register_buffer ensures it moves with .to(device)
        self.register_buffer("current_content", torch.zeros(1, dim))
        self.current_source: str = "None"
        self.current_salience: float = 0.0
        
        # Logs
        self.history: List[Dict[str, Any]] = []

    def upload_to_workspace(self, source_name: str, content: Dict[str, torch.Tensor], salience: float):
        """
        各モジュールが情報をワークスペースに提示する。
        """
        # 確率的スイッチング
        prob_switch = torch.sigmoid(torch.tensor(salience - self.current_salience)).item()
        
        if salience > self.current_salience or (torch.rand(1).item() < prob_switch * 0.1):
            features = content.get("features")
            if features is not None:
                self._update_content(features)
                self.current_source = source_name
                self.current_salience = salience

    def _update_content(self, input_tensor: torch.Tensor):
        """入力テンソルをワークスペースの次元に合わせて更新"""
        
        # --- Device Safety Check ---
        # 入力が内部バッファと異なるデバイスにある場合、強制的に合わせる
        if input_tensor.device != self.current_content.device:
            input_tensor = input_tensor.to(self.current_content.device)
        # ---------------------------

        batch_size = input_tensor.shape[0]
        target_dim = self.dim
        
        # 簡易的な次元調整
        if input_tensor.shape[-1] > target_dim:
            # 圧縮 (Truncate)
            processed = input_tensor[:, :target_dim]
        elif input_tensor.shape[-1] < target_dim:
            # 拡張 (Zero Padding)
            padding = torch.zeros(batch_size, target_dim - input_tensor.shape[-1], device=input_tensor.device)
            processed = torch.cat([input_tensor, padding], dim=1)
        else:
            processed = input_tensor
            
        # 時間的平滑化
        self.current_content = self.current_content * self.decay + processed * (1 - self.decay)

    def get_current_thought(self) -> torch.Tensor:
        """現在意識に上っている内容を取得"""
        return self.current_content

    def step(self):
        """時間経過による減衰"""
        self.current_salience *= 0.95
        if self.current_salience < 0.1:
            self.current_source = "None"