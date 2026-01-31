# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace Module v2.5 (Fix: Sequence Pooling)
# 目的・内容:
#   - _update_content メソッドで、入力が3次元以上（シーケンスや画像）の場合に
#     平均プーリングを行って (Batch, Dim) に縮約するように修正。
#   - これにより可変長の入力に対して形状不整合を防ぐ。

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

class GlobalWorkspace(nn.Module):
    """
    Implementation of Global Workspace Theory with Metadata support.
    """
    current_content: torch.Tensor

    def __init__(self, dim: int = 64, decay: float = 0.9):
        super().__init__()
        self.dim = dim
        self.decay = decay

        # Current conscious content
        self.register_buffer("current_content", torch.zeros(1, dim))
        self.current_source: str = "None"
        self.current_salience: float = 0.0
        
        # メタデータの保持用
        self.current_metadata: Dict[str, Any] = {}

        self.memory_buffers: Dict[str, Any] = {}
        self.subscribers: List[Any] = []

    def subscribe(self, callback: Any):
        self.subscribers.append(callback)

    def upload_to_workspace(self, source_name: str, content: Dict[str, Any], salience: float, **kwargs: Any):
        """
        各モジュールからの情報をワークスペースへアップロード。
        競合（Competition）を行い、勝った情報を放送する。
        """
        # 簡易的な確率的スイッチング (Salienceベース)
        prob_switch = torch.sigmoid(torch.tensor(salience - self.current_salience)).item()
        should_switch = salience > self.current_salience or (torch.rand(1).item() < prob_switch * 0.2)

        if should_switch:
            features = content.get("features")
            if features is not None:
                self._update_content(features)
                self.current_source = source_name
                self.current_salience = salience
                self.current_metadata = content

    def _update_content(self, input_tensor: torch.Tensor):
        if input_tensor.device != self.current_content.device:
            input_tensor = input_tensor.to(self.current_content.device)

        # [Fix] シーケンス次元/空間次元の縮約 (Batch, Seq, Dim) -> (Batch, Dim)
        # 意識は「要約」された情報を保持するため、時間/空間方向は平均化する
        if input_tensor.dim() > 2:
            # 最後の次元(特徴量)を残して、それ以外(1番目〜最後から2番目)を平均
            # 例: (32, 784, 256) -> (32, 256)
            # 例: (32, 28, 28, 256) -> (32, 256)
            flattened = input_tensor.view(input_tensor.shape[0], -1, input_tensor.shape[-1])
            input_tensor = flattened.mean(dim=1)

        batch_size = input_tensor.shape[0]
        # 次元合わせ
        if input_tensor.shape[-1] != self.dim:
            if input_tensor.shape[-1] > self.dim:
                processed = input_tensor[..., :self.dim]
            else:
                padding = torch.zeros(
                    *input_tensor.shape[:-1], self.dim - input_tensor.shape[-1], 
                    device=input_tensor.device
                )
                processed = torch.cat([input_tensor, padding], dim=-1)
        else:
            processed = input_tensor

        # Batchサイズが異なる場合（現在の保持内容が1で、入力が32の場合など）
        # current_contentを拡張して対応（ブロードキャスト）
        if self.current_content.shape[0] != batch_size:
            if self.current_content.shape[0] == 1:
                # (1, dim) -> (Batch, dim)
                self.current_content = self.current_content.expand(batch_size, -1).clone()
            elif batch_size == 1:
                # (Batch, dim) -> (1, dim) 平均化して縮小
                self.current_content = self.current_content.mean(dim=0, keepdim=True)
            else:
                # どちらも1でない場合はサイズが合わないため、リセットして新しいバッチに合わせる
                self.current_content = torch.zeros(batch_size, self.dim, device=self.current_content.device)

        # 慣性項つき更新 (急激な変化を抑制)
        self.current_content = self.current_content * self.decay + processed * (1 - self.decay)

    def step(self):
        """時間ステップごとの減衰処理"""
        self.current_salience *= 0.95
        if self.current_salience < 0.1:
            self.current_source = "None"
            self.current_metadata = {}

    def get_current_content(self) -> Dict[str, Any]:
        """
        現在の意識内容を辞書形式で返す。
        """
        return {
            "features": self.current_content,
            "source": self.current_source,
            "salience": self.current_salience,
            **self.current_metadata
        }

    def get_current_thought(self) -> torch.Tensor:
        """ベクトルのみを取得するエイリアス"""
        return self.current_content

    def reset(self):
        # Reset to (1, dim) zero tensor
        self.current_content = torch.zeros(1, self.dim, device=self.current_content.device)
        self.current_source = "None"
        self.current_salience = 0.0
        self.current_metadata = {}