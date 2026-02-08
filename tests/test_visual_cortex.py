# directory: snn_research/models/bio
# filename: visual_cortex.py
# description: SARAエンジン統合型視覚野モデル (Fix: LIFNeuron初期化引数修正)

import torch
import torch.nn as nn
from typing import Dict, Any, List

# SARAアダプターのインポート (IT野の記憶・認識用)
from snn_research.models.adapters.sara_adapter import SaraAdapter

# LIFニューロン (プロジェクト標準のものを使用)
# もし標準モジュールに依存できない場合は、ここで互換クラスを定義するが、
# 今回のエラーは引数不足なので、標準クラスが想定されていると推測される。
# ここでは安全のため、引数を受け取れる互換LIFを定義する。
try:
    from snn_research.core.neurons.lif_neuron import LIFNeuron
except ImportError:
    class LIFNeuron(nn.Module):
        def __init__(self, features=None, tau=2.0, threshold=1.0):
            super().__init__()
            self.tau = tau
            self.threshold = threshold
        def forward(self, x):
            return (x > self.threshold).float()

class BioVisualCortex(nn.Module):
    """
    Bio-Inspired Visual Cortex Model (v2.1):
    - V1/V2 (Early Vision): Spiking CNN (Feature Extraction)
    - V4/IT (High-level Vision): SARA Engine (Object Recognition & Memory)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        in_channels = config.get("in_channels", 3)
        # Base channel count (e.g. 32)
        base_channels = config.get("base_channels", 32) 
        
        # --- V1: Early Visual Processing ---
        self.v1_conv = nn.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=5, stride=2, padding=2)
        # 修正: LIFNeuronにfeatures引数(チャネル数)を渡す
        self.v1_lif = LIFNeuron(features=base_channels)
        
        # --- V2: Intermediate Processing ---
        v2_channels = base_channels * 2
        self.v2_conv = nn.Conv2d(in_channels=base_channels, out_channels=v2_channels, kernel_size=3, stride=2, padding=1)
        # 修正: LIFNeuronにfeatures引数(チャネル数)を渡す
        self.v2_lif = LIFNeuron(features=v2_channels)
        
        # 特徴マップのフラット化後の次元計算
        img_size = config.get("image_size", 28)
        # 2回ダウンサンプリング (stride 2) -> size / 4
        feat_size = img_size // 4
        feature_dim = v2_channels * feat_size * feat_size
        
        # --- IT Cortex: Object Recognition & Memory (SARA Engine) ---
        sara_config = {
            "input_size": feature_dim,
            "hidden_size": 1024,
            "output_size": config.get("num_classes", 10),
            "enable_rlm": config.get("enable_rlm", True),
            "enable_attractor": config.get("enable_attractor", True)
        }
        
        self.it_cortex = SaraAdapter(sara_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        顺伝播: Retina -> V1 -> V2 -> IT (SARA) -> Perception
        Args:
            x: [Batch, Channels, Height, Width] or [Batch, Time, C, H, W]
        """
        if x.dim() == 5: # [B, T, C, H, W]
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            time_distributed = True
        else:
            time_distributed = False
            b = x.shape[0]

        # --- V1 Area ---
        v1_pot = self.v1_conv(x)
        v1_spk = self.v1_lif(v1_pot)
        
        # --- V2 Area ---
        v2_pot = self.v2_conv(v1_spk)
        v2_spk = self.v2_lif(v2_pot)
        
        # Flatten
        features = v2_spk.view(v2_spk.size(0), -1)
        
        # 時間次元の復元
        if time_distributed:
            features = features.view(b, t, -1)
            # SARAへの入力用に時間平均
            features_sara_input = features.mean(dim=1)
        else:
            features_sara_input = features

        # --- IT Area (SARA Engine) ---
        perception = self.it_cortex(features_sara_input)
        
        return perception

    def reset(self):
        """内部状態のリセット"""
        # ニューロンのリセットがあれば呼ぶ
        if hasattr(self.v1_lif, "reset"): self.v1_lif.reset()
        if hasattr(self.v2_lif, "reset"): self.v2_lif.reset()
        # SARAエンジンのリセット
        # (アダプター経由で呼べる場合、または再生成)
        pass

# 互換エイリアス
VisualCortex = BioVisualCortex