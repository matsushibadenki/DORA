# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace Module v2.3 (Type Safe)
# 目的・内容:
#   nn.Moduleとの名前空間の衝突を回避し、型安全性を向上。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory with OS-like Shared Memory support.
    """

    def __init__(self, dim: int = 64, decay: float = 0.9):
        super().__init__()
        self.dim = dim
        self.decay = decay

        # Current conscious content (The "Thought")
        self.current_content: torch.Tensor
        self.register_buffer("current_content", torch.zeros(1, dim))
        self.current_source: str = "None"
        self.current_salience: float = 0.0

        # OS-like Shared Memory Buffers (Renamed to avoid conflict with nn.Module.buffers)
        self.memory_buffers: Dict[str, Any] = {}

        # Logs
        self.history: List[Dict[str, Any]] = []
        self.subscribers: List[Any] = []

    def subscribe(self, callback: Any):
        """Register a callback for conscious broadcasts."""
        self.subscribers.append(callback)

    def write(self, key: str, value: Any, salience: float = 1.0):
        """
        [OS API] 指定したキーでデータをバッファに書き込む。
        """
        self.memory_buffers[key] = value
        
        # 意識へのアップロード
        content_to_upload = {}
        if isinstance(value, torch.Tensor):
            content_to_upload = {"features": value}
        elif isinstance(value, dict):
            content_to_upload = value
        
        if content_to_upload:
            self.upload_to_workspace(key, content_to_upload, salience)

    def read(self, key: str) -> Any:
        """
        [OS API] 指定したキーのデータをバッファから読み込む。
        """
        return self.memory_buffers.get(key, None)

    def get_current_content(self) -> torch.Tensor:
        """Alias for get_current_thought."""
        return self.current_content

    def publish(self, content: Any):
        """Test compatibility method."""
        if isinstance(content, str):
            tensor_content = torch.randn(
                1, self.dim, device=self.current_content.device)
        elif isinstance(content, torch.Tensor):
            tensor_content = content
        else:
            tensor_content = torch.randn(
                1, self.dim, device=self.current_content.device)

        self._update_content(tensor_content)
        self.current_salience = 1.0

    def upload_to_workspace(self, source_name: str, content: Dict[str, Any], salience: float, **kwargs: Any):
        """
        各モジュールが情報をワークスペースに提示する。
        """
        if 'source' in kwargs:
            source_name = kwargs['source']
        if 'data' in kwargs:
            content = kwargs['data']

        # 確率的スイッチング
        prob_switch = torch.sigmoid(torch.tensor(
            salience - self.current_salience)).item()

        if salience > self.current_salience or (torch.rand(1).item() < prob_switch * 0.1):
            features = content.get("features")
            if features is not None:
                self._update_content(features)
                self.current_source = source_name
                self.current_salience = salience

    def _update_content(self, input_tensor: torch.Tensor):
        """入力テンソルをワークスペースの次元に合わせて更新"""
        if input_tensor.device != self.current_content.device:
            input_tensor = input_tensor.to(self.current_content.device)

        batch_size = input_tensor.shape[0]
        target_dim = self.dim

        if input_tensor.shape[-1] > target_dim:
            processed = input_tensor[:, :target_dim]
        elif input_tensor.shape[-1] < target_dim:
            padding = torch.zeros(
                batch_size, target_dim - input_tensor.shape[-1], device=input_tensor.device)
            processed = torch.cat([input_tensor, padding], dim=1)
        else:
            processed = input_tensor

        self.current_content = self.current_content * \
            self.decay + processed * (1 - self.decay)

    def get_current_thought(self) -> torch.Tensor:
        return self.current_content

    def step(self):
        self.current_salience *= 0.95
        if self.current_salience < 0.1:
            self.current_source = "None"

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Process inputs from multiple modules and broadcast the most salient one.
        """
        saliences = []
        for name, tensor in inputs.items():
            salience = tensor.norm().item()
            saliences.append(salience)
            self.write(name, tensor, salience)

        self.step()

        return {
            "winner": self.current_source,
            "broadcast": self.current_content,
            "salience": saliences
        }