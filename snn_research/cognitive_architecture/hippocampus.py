# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampus (Short-term Episodic Memory) with Flushing
# Description: 短期記憶を管理し、睡眠時に長期記憶へ転送(flush)するための機能を提供。

import torch
import torch.nn as nn
import time
from typing import List, Dict, Any, Optional

class Hippocampus(nn.Module):
    """
    短期エピソード記憶を保持するモジュール。
    一定容量を超えると古い記憶から忘却するが、flush_memories()で全て取り出し可能。
    """
    def __init__(self, capacity: int = 200, input_dim: int = 64, device: str = "cpu", **kwargs):
        """
        Args:
            capacity (int): 保持できる最大エピソード数
            input_dim (int): 入力ベクトルの次元
            device (str): デバイス設定
            **kwargs: 互換性用の引数 (short_term_capacityなど)
        """
        super().__init__()
        # 互換性: short_term_capacity が渡された場合は capacity として扱う
        if "short_term_capacity" in kwargs:
            capacity = kwargs["short_term_capacity"]
            
        self.capacity = capacity
        self.input_dim = input_dim
        self.device = device
        
        # エピソードバッファ (FIFOキューとして動作)
        self.episodic_buffer: List[Dict[str, Any]] = []
        
        # 統計情報
        self.total_memories_formed = 0

    def process(self, memory_item: Dict[str, Any]):
        """
        新しい記憶を追加する。
        Args:
            memory_item: {"embedding": Tensor, "text": str, "timestamp": float, ...}
        """
        # 必須キーの確認と補完
        if "timestamp" not in memory_item:
            memory_item["timestamp"] = time.time()
            
        # バッファに追加
        self.episodic_buffer.append(memory_item)
        self.total_memories_formed += 1
        
        # 容量オーバー時の処理 (古いものを削除)
        if len(self.episodic_buffer) > self.capacity:
            removed = self.episodic_buffer.pop(0)
            # ログ出力したければここで
            
    def retrieve(self, query_vector: torch.Tensor, k: int = 1) -> List[Dict[str, Any]]:
        """
        (簡易実装) コサイン類似度などで検索するロジックのプレースホルダー。
        現状は直近の記憶を返すだけ。
        """
        if not self.episodic_buffer:
            return []
        
        # 本来はベクトル検索だが、デモ用に最新k件を返す
        return self.episodic_buffer[-k:]

    def flush_memories(self) -> List[Dict[str, Any]]:
        """
        [Fix] 保持している全ての短期記憶を返し、バッファをクリアする。
        睡眠時の記憶固定化(Consolidation)で使用される。
        """
        memories = list(self.episodic_buffer) # コピーを作成
        self.episodic_buffer.clear()          # バッファを空にする
        return memories

    def clear(self):
        self.episodic_buffer.clear()

    def forward(self, x):
        # nn.Moduleとしてのダミー
        return x