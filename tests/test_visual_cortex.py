# ファイルパス: tests/test_visual_cortex.py
# 日本語タイトル: 視覚野モデル (VisualCortex) 単体テスト v2.2
# 目的・内容: 
#   VisualCortexの形状不一致エラーを修正し、静止画・動画入力に対する動作を検証。

import unittest
import torch
from snn_research.models.bio.visual_cortex import VisualCortex

class TestVisualCortex(unittest.TestCase):
    def setUp(self):
        # テスト用の設定
        self.in_channels = 3
        self.base_channels = 16 
        self.time_steps = 5
        self.neuron_params = {"tau_mem": 20.0, "base_threshold": 1.0}

    def test_visual_cortex_static_image(self):
        """静止画入力(32x32)に対する視覚野の動作テスト"""
        # 入力画像サイズに合わせて input_shape を指定
        model = VisualCortex(
            input_shape=(32, 32),
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            time_steps=self.time_steps,
            neuron_params=self.neuron_params
        )
        
        # (Batch, Channel, H, W)
        x = torch.randn(2, 3, 32, 32)
        
        # モデルは (B, T, Features) を返す
        output = model(x)
        
        # 出力形状の確認
        self.assertEqual(output.dim(), 3)
        self.assertEqual(output.shape[0], 2) # Batch
        self.assertEqual(output.shape[1], self.time_steps) # Time
        # Output Dim = base_channels * 8 (IT layer)
        self.assertEqual(output.shape[2], self.base_channels * 8)

    def test_visual_cortex_video_stream(self):
        """動画ストリーム(32x32)に対する動作テスト"""
        model = VisualCortex(
            input_shape=(32, 32),
            in_channels=1, # モノクロ動画
            base_channels=self.base_channels,
            time_steps=self.time_steps, 
            neuron_params=self.neuron_params
        )
        
        # (Batch, Time, Channel, H, W)
        x = torch.randn(2, 8, 1, 32, 32)
        
        output = model(x)
        
        # 出力の時間次元が入力(8)と一致することを確認
        self.assertEqual(output.shape[1], 8)
        self.assertEqual(output.shape[0], 2)

    def test_reset(self):
        """内部状態のリセット機能のテスト (入力サイズ 16x16)"""
        # 入力画像サイズに合わせて input_shape を指定
        model = VisualCortex(
            input_shape=(16, 16),
            in_channels=self.in_channels,
            base_channels=self.base_channels,
            time_steps=self.time_steps,
            neuron_params=self.neuron_params
        )
        
        # 決定論的な動作を保証
        model.eval()
        
        x = torch.randn(1, 3, 16, 16)
        
        # 1回目の実行
        out1 = model(x)
        
        # リセット
        model.reset_state()
        
        # 2回目の実行
        out2 = model(x)
        
        # 出力が一致することを確認
        self.assertTrue(torch.allclose(out1, out2), "Output mismatch after reset in eval mode")

if __name__ == "__main__":
    unittest.main()