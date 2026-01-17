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

    def __init__(self, output_dim: int = 784, device: str = 'cpu', **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.device = device
        self.kwargs = kwargs  # Store extra args
        self.time_steps = kwargs.get('time_steps', 1)
        logger.info(
            f"🌐 UniversalEncoder initialized (Offline Mode). Args: {kwargs}")

    def forward(self, x, modality: str = "text", **kwargs):
        # ダミー実装: ランダムなスパイクを返す
        batch_size = 1
        if isinstance(x, torch.Tensor):
            batch_size = x.shape[0]

        return (torch.rand(batch_size, self.time_steps, self.output_dim) > 0.95).float().to(self.device)

    # Alias for compatibility with Brain v4 and Agents
    encode = forward
    encode_text = forward


# Legacy support / Alias
UniversalSpikeEncoder = UniversalEncoder
