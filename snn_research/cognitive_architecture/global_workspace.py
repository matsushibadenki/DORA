# ファイルパス: snn_research/cognitive_architecture/global_workspace.py
# 日本語タイトル: Global Workspace (Consciousness Hub)
# 目的・内容:
#   脳内の情報の「競合」と「放送（Broadcast）」を管理するモジュール。
#   Neuromorphic OSの設計方針に基づき、これは中央制御装置ではなく
#   「情報共有バス」として機能する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Any, Optional, List, Callable

logger = logging.getLogger(__name__)

class GlobalWorkspace(nn.Module):
    """
    Global Workspace Theory (GWT) に基づく情報共有ハブ。
    
    機能:
    1. 各モジュールからの入力を受け取る
    2. 注意機構（Attention）による競合（Competition）を行う
    3. 勝者の情報を全モジュールに放送（Broadcast）する
    """
    workspace_state: torch.Tensor

    def __init__(
        self,
        dim: int = 64,
        num_slots: int = 1,
        decay: float = 0.9,
        model_registry: Optional[Any] = None
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.decay = decay
        self.model_registry = model_registry

        # 意識の内容（Global Working Memory）
        # 揮発性であり、常に減衰または更新される
        self.register_buffer("workspace_state", torch.zeros(1, dim))

        # Attention Mechanism (Salience Selector)
        # 入力の重要度を判定する簡易ネットワーク
        self.selector = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1)
        )

        # Subscribers (放送を受け取るモジュール群)
        self.subscribers: List[Callable[[str, Any], None]] = []
        self.current_content: Dict[str, Any] = {}

        logger.info(f"👁️ Global Workspace (Consciousness) initialized (Dim: {dim}).")

    def subscribe(self, callback: Callable[[str, Any], None]):
        """他のモジュールが意識の放送を受信するために登録する"""
        self.subscribers.append(callback)

    def upload_to_workspace(self, source: str, data: Any, salience: float = 0.5):
        """
        モジュールからワークスペースへの情報提供。
        一定以上のSalience（顕著性）があればワークスペースの状態を書き換える。
        """
        if salience > 0.7:
            if isinstance(data, dict) and "vector_state" in data:
                vec = data["vector_state"]
            elif isinstance(data, dict) and "features" in data:
                vec = data["features"]
            else:
                vec = None

            if isinstance(vec, torch.Tensor):
                # 次元合わせと更新
                if vec.dim() == 1:
                     vec = vec.unsqueeze(0)
                
                # 特徴量の次元が合う場合のみ更新（本来はProjectorが必要）
                if vec.shape[-1] == self.dim:
                    self.workspace_state = vec.detach()

            self._broadcast_to_subscribers(source, data)

    def broadcast(self, inputs: List[Any], context: Optional[str] = None) -> Any:
        """レガシーインターフェース互換用"""
        tensor_inputs = {}
        for i, item in enumerate(inputs):
            if isinstance(item, torch.Tensor):
                tensor_inputs[f"input_{i}"] = item
            elif isinstance(item, dict) and "features" in item:
                tensor_inputs[f"module_{i}"] = item["features"]

        if tensor_inputs:
            result = self.forward(tensor_inputs)
            if result["winner"] is not None:
                self._broadcast_to_subscribers(
                    str(result["winner"]), result["broadcast"])
            return result["broadcast"]

        return self.workspace_state

    def get_current_thought(self) -> torch.Tensor:
        """現在の意識状態ベクトルを取得"""
        return self.workspace_state

    def get_information(self) -> torch.Tensor:
        """テスト互換用"""
        return self.get_current_thought()

    def get_current_content(self) -> Dict[str, Any]:
        """現在の意識内容の詳細を取得（可説明性用）"""
        return self.current_content

    def _broadcast_to_subscribers(self, source: str, content: Any):
        # コンテンツの保持
        if isinstance(content, dict):
            self.current_content = content
        else:
            self.current_content = {"type": "raw", "data": content, "source": source}

        # 登録者全員に通知
        for callback in self.subscribers:
            try:
                callback(source, content)
            except Exception as e:
                logger.warning(f"Broadcast error: {e}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        競合プロセスを実行する。
        """
        candidates = []
        names = []

        for name, tensor in inputs.items():
            flat_tensor = tensor
            
            # 時間次元の集約 (Batch, Time, Dim) -> (Batch, Dim)
            if flat_tensor.dim() > 2:
                flat_tensor = flat_tensor.mean(dim=1)
            
            # [Fix] 1次元入力 (Dim,) の場合、バッチ次元を追加して (1, Dim) にする
            if flat_tensor.dim() == 1:
                flat_tensor = flat_tensor.unsqueeze(0)

            # 次元調整 (Feature Dim mismatch handling)
            current_dim = flat_tensor.shape[-1]
            if current_dim != self.dim:
                if current_dim < self.dim:
                    # パディング
                    pad = self.dim - current_dim
                    flat_tensor = F.pad(flat_tensor, (0, pad))
                else:
                    # 切り捨て
                    flat_tensor = flat_tensor[:, :self.dim]

            candidates.append(flat_tensor)
            names.append(name)

        if not candidates:
            return {"broadcast": self.workspace_state, "winner": None, "salience": None}

        # 候補をスタックして一括評価
        stack = torch.cat(candidates, dim=0) # (TotalBatch, Dim)
        
        scores = self.selector(stack).squeeze(-1) # (TotalBatch,)
        
        # 数値安定性と探索のためのノイズ
        noise = torch.randn_like(scores) * 0.1
        probs = F.softmax(scores + noise, dim=0)

        # 勝者決定
        winner_idx = int(torch.argmax(probs).item())
        winner_name = names[winner_idx]
        winner_content = candidates[winner_idx]

        # 状態更新 (Exponential Moving Average)
        if winner_content.shape[0] != self.workspace_state.shape[0]:
             if winner_content.shape[0] == 1:
                 winner_content = winner_content.expand_as(self.workspace_state)

        new_state = (1 - self.decay) * winner_content + \
            self.decay * self.workspace_state
        self.workspace_state = new_state.detach()

        return {
            "broadcast": new_state,
            "winner": winner_name,
            "salience": probs.detach()
        }