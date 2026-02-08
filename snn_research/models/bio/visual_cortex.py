# directory: snn_research/models/bio
# filename: visual_cortex.py
# description: SARAエンジン統合型視覚野モデル (V1/V2: Spiking CNN -> IT: SARA Attractor)

import torch
import torch.nn as nn
from typing import Dict, Any, List

# SARAアダプターのインポート (IT野の記憶・認識用)
from snn_research.models.adapters.sara_adapter import SaraAdapter

# LIFニューロン (局所定義またはインポート)
try:
    from snn_research.core.neurons.lif_neuron import LIFNeuron
except ImportError:
    class LIFNeuron(nn.Module):
        def __init__(self): super().__init__(); 
        def forward(self, x): return (x > 1.0).float()

class BioVisualCortex(nn.Module):
    """
    Bio-Inspired Visual Cortex Model (v2.0 SARA Integrated):
    
    Structure:
    - Retina: 入力エンコーディング (Pixel -> Spike)
    - V1/V2 (Early Vision): Spiking CNNによる特徴抽出 (エッジ、テクスチャ)
    - V4/IT (High-level Vision): SARAエンジンによる物体認識とアトラクタ記憶
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # --- V1/V2: Early Visual Processing (Feature Extraction) ---
        # 畳み込みSNN層
        self.v1_conv = nn.Conv2d(in_channels=config.get("in_channels", 3), out_channels=32, kernel_size=5, stride=2, padding=2)
        self.v1_lif = LIFNeuron()
        
        self.v2_conv = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.v2_lif = LIFNeuron()
        
        # 特徴マップのフラット化後の次元計算 (簡易計算: 2回ダウンサンプリング)
        # 例: 28x28 -> 14x14 -> 7x7. 64ch * 7 * 7 = 3136
        img_size = config.get("image_size", 28)
        feature_dim = 64 * (img_size // 4) * (img_size // 4)
        
        # --- IT Cortex: Object Recognition & Memory (SARA Engine) ---
        # ここでSARAエンジンを使用して、抽出された視覚特徴を長期記憶と照合する
        sara_config = {
            "input_size": feature_dim,
            "hidden_size": 1024,
            "output_size": config.get("num_classes", 10),
            "enable_rlm": True,       # 報酬学習有効
            "enable_attractor": True  # アトラクタ記憶有効
        }
        
        # 既存のSARA実装を利用 (アダプター経由)
        self.it_cortex = SaraAdapter(sara_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播: Retina -> V1 -> V2 -> IT (SARA) -> Perception
        Args:
            x: [Batch, Channels, Height, Width] (Static Image) 
               or [Batch, Time, C, H, W] (Video/Spike Stream)
        """
        # 時間次元のハンドリング (静止画なら時間次元追加、動画ならループ)
        # ここでは静止画入力を想定したRate Coding的な一括処理、または時間平均
        
        if x.dim() == 5: # [B, T, C, H, W]
            # 時間次元をバッチに統合してCNNを通す (Batch*Time, C, H, W)
            b, t, c, h, w = x.shape
            x = x.view(b * t, c, h, w)
            time_distributed = True
        else:
            time_distributed = False

        # --- V1 Area ---
        v1_pot = self.v1_conv(x)
        v1_spk = self.v1_lif(v1_pot)
        
        # --- V2 Area ---
        v2_pot = self.v2_conv(v1_spk)
        v2_spk = self.v2_lif(v2_pot)
        
        # Flatten
        features = v2_spk.view(v2_spk.size(0), -1)
        
        # 時間次元の復元 (SARAに入力するため)
        if time_distributed:
            features = features.view(b, t, -1)
            # SARAが [Batch, Dim] を期待する場合、時間平均を取るか、
            # アダプターがシーケンス対応ならそのまま渡す。
            # SARA Adapterは通常 [Batch, Dim] なので平均化する
            features_sara_input = features.mean(dim=1)
        else:
            features_sara_input = features

        # --- IT Area (SARA Engine) ---
        # 視覚特徴から概念/物体を想起
        perception = self.it_cortex(features_sara_input)
        
        return perception

    def get_feature_maps(self, x):
        """デバッグ/可視化用: V1, V2の内部活性を取得"""
        with torch.no_grad():
            v1 = self.v1_lif(self.v1_conv(x))
            v2 = self.v2_lif(self.v2_conv(v1))
        return {"v1": v1, "v2": v2}

# 旧クラス名の互換性維持 (VisualCortexとしてインポートされる場合に対応)
VisualCortex = BioVisualCortex