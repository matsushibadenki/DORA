# ファイルパス: snn_research/io/universal_encoder.py
# 日本語タイトル: Universal Encoder (Offline Fix)
# 修正内容: インポート時の自動ダウンロードを無効化。

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

# Transformersを無効化
TRANSFORMERS_AVAILABLE = False

class UniversalEncoder(nn.Module):
    """
    マルチモーダル入力（テキスト、画像、音声）を統一的なスパイク表現に変換するエンコーダ。
    (現在はオフラインモード/プレースホルダーとして機能)
    """
    def __init__(self, output_dim: int = 784, device: str = 'cpu'):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        logger.info("🌐 UniversalEncoder initialized (Offline Mode).")

    def forward(self, x, modality: str = "text"):
        # ダミー実装: ランダムなスパイクを返す
        batch_size = 1
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]
        
        return torch.rand(batch_size, self.output_dim).to(self.device)