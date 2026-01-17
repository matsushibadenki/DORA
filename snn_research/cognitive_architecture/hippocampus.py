# ファイルパス: snn_research/cognitive_architecture/hippocampus.py
# 日本語タイトル: Hippocampal Episodic Buffer
# 目的・内容:
#   短期的なエピソード記憶を保持し、睡眠時や休息時に
#   「リプレイ（Replay）」として大脳皮質へ信号を送り返すモジュール。

import torch
import torch.nn as nn
import random
from typing import List, Optional, Tuple
from collections import deque

class Hippocampus(nn.Module):
    """
    Hippocampal Episodic Memory System.
    
    役割:
    1. Encoding: 感覚入力（V1/Association）のパターンを一時的に保持する。
    2. Replay: 保持したパターンをノイズ付きで再生成し、皮質の学習を促進する。
    """
    def __init__(self, capacity: int = 200, input_dim: int = 784, device: str = "cpu"):
        super().__init__()
        self.capacity = capacity
        self.device = device
        
        # エピソードバッファ (FIFO)
        # 実際の海馬はニューラルネットワークだが、ここでは機能モデルとしてdequeを使用
        self.episodic_buffer: deque = deque(maxlen=capacity)
        
    def store_episode(self, pattern: torch.Tensor):
        """
        覚醒時の経験（パターン）を保存する。
        入力が十分に強い（意味がある）場合のみ保存するフィルタリングを行う。
        """
        # バッチ次元を落として保存
        if pattern.sum() > 0.1: # 無音/無信号は無視
            detached_pattern = pattern.detach().cpu()
            self.episodic_buffer.append(detached_pattern)

    def generate_replay(self, batch_size: int = 1, noise_level: float = 0.05) -> Optional[torch.Tensor]:
        """
        睡眠時・休息時に記憶をランダムに再生する（Sharp Wave Ripples）。
        """
        if len(self.episodic_buffer) < 1:
            return None
            
        # ランダムサンプリング
        episodes = random.sample(self.episodic_buffer, k=min(len(self.episodic_buffer), batch_size))
        
        # テンソル化してデバイスへ転送
        replay_batch = torch.stack(episodes).to(self.device)
        
        # ノイズ付与（汎化性能の向上と、夢の不確実性の表現）
        noise = torch.randn_like(replay_batch) * noise_level
        
        return replay_batch + noise

    def clear_memory(self):
        """完全な忘却"""
        self.episodic_buffer.clear()

    def get_memory_stat(self) -> dict:
        return {
            "stored_episodes": len(self.episodic_buffer),
            "capacity": self.capacity
        }