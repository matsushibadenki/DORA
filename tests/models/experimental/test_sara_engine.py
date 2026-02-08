# directory: tests/models/experimental
# file: test_sara_engine.py
# purpose: SARAエンジンの各モジュールおよび統合動作の単体テスト
# author: System
# date: 2026-02-08

import unittest
import torch
import sys
import os

# プロジェクトルートへのパスを追加してインポート可能にする
# (実行ディレクトリ構成に合わせて適宜調整が必要です)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.models.experimental.sara_engine import (
        SARAEngine, 
        SNNEncoder, 
        LegendreSpikeAttractor, 
        RecursiveMeaningLayer, 
        RecursionController, 
        SNNDecoder
    )
except ImportError:
    # パス解決がうまくいかない場合のフォールバック（同ディレクトリにある場合など）
    try:
        from sara_engine import (
            SARAEngine, SNNEncoder, LegendreSpikeAttractor, 
            RecursiveMeaningLayer, RecursionController, SNNDecoder
        )
    except ImportError:
        print("Error: sara_engine.py not found. Please ensure the file is in the python path.")
        sys.exit(1)

class TestSARAEngine(unittest.TestCase):
    def setUp(self):
        """テストケースごとの初期化"""
        # テスト用の共通パラメータ
        self.input_dim = 10
        self.n_encode_neurons = 32
        self.d_legendre = 12
        self.d_meaning = 64
        self.n_output_neurons = 5
        self.batch_size = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # エンジンのインスタンス化
        self.engine = SARAEngine(
            input_dim=self.input_dim,
            n_encode_neurons=self.n_encode_neurons,
            d_legendre=self.d_legendre,
            d_meaning=self.d_meaning,
            n_output_neurons=self.n_output_neurons
        ).to(self.device)

    def test_snn_encoder_output_shape_and_values(self):
        """SNNエンコーダの出力形状とバイナリ値の検証"""
        encoder = self.engine.encoder
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        time_steps = 20
        
        spikes = encoder(x, time_steps=time_steps)
        
        # 形状チェック: (Batch, Time, Neurons)
        self.assertEqual(spikes.shape, (self.batch_size, time_steps, self.n_encode_neurons))
        
        # 値チェック: 0または1のみであることを確認
        unique_values = torch.unique(spikes)
        for v in unique_values:
            self.assertIn(v.item(), [0.0, 1.0])

    def test_legendre_attractor_dynamics(self):
        """Legendreアトラクタ層の動作検証"""
        legendre = self.engine.legendre
        time_steps = 30
        # ダミーのスパイク入力
        spikes = torch.randint(0, 2, (self.batch_size, time_steps, self.n_encode_neurons)).float().to(self.device)
        
        m = legendre(spikes)
        
        # 形状チェック: (Batch, d_legendre)
        self.assertEqual(m.shape, (self.batch_size, self.d_legendre))
        
        # 状態がゼロでないことを確認（入力があれば状態が変化するはず）
        self.assertFalse(torch.all(m == 0), "Legendre coefficients should not be all zero for random input")
        
        # 勾配計算が可能かチェック
        m.sum().backward()

    def test_rlm_recursion_depth_control(self):
        """RLM層の再帰動作と深度制御の検証"""
        rlm = self.engine.rlm
        m = torch.randn(self.batch_size, self.d_legendre).to(self.device)
        
        # 1. 強制的に深度制限を設けてテスト
        max_depth = 5
        z, depth, trajectory = rlm(m, max_depth=max_depth)
        
        # 出力形状チェック
        self.assertEqual(z.shape, (self.batch_size, self.d_meaning))
        
        # 深度チェック
        self.assertLessEqual(depth, max_depth)
        self.assertEqual(len(trajectory), depth + 1) # 初期状態 z0 + depth回更新

    def test_recursion_controller_logic(self):
        """再帰制御器の停止判定ロジック検証"""
        controller = RecursionController(epsilon=0.1, max_depth=10, energy_budget=100)
        
        z_prev = torch.zeros(1, 10)
        z_curr_converged = torch.zeros(1, 10) + 0.01 # 収束している
        z_curr_diverged = torch.zeros(1, 10) + 1.0   # 収束していない
        
        # 収束ケース
        should_stop = controller.should_stop(z_prev, z_curr_converged, depth=5, energy_used=10)
        self.assertTrue(should_stop, "Should stop when converged")
        
        # 未収束ケース
        should_stop = controller.should_stop(z_prev, z_curr_diverged, depth=5, energy_used=10)
        self.assertFalse(should_stop, "Should not stop when diverged and budget exists")
        
        # エネルギー切れケース
        should_stop = controller.should_stop(z_prev, z_curr_diverged, depth=5, energy_used=101)
        self.assertTrue(should_stop, "Should stop when energy budget exceeded")

    def test_full_engine_integration(self):
        """統合エンジンのEnd-to-End推論検証"""
        x = torch.randn(self.batch_size, self.input_dim).to(self.device)
        
        # 推論実行
        output_spikes, depth, uncertainty = self.engine(x)
        
        # 出力スパイクチェック
        # (Batch, Time(default 100), OutputNeurons)
        self.assertEqual(output_spikes.shape[0], self.batch_size)
        self.assertEqual(output_spikes.shape[2], self.n_output_neurons)
        
        # メタデータチェック
        self.assertIsInstance(depth, int)
        self.assertIsInstance(uncertainty, float)
        self.assertGreater(depth, 0)
        
        print(f"\n[Info] Inference Depth: {depth}, Uncertainty: {uncertainty:.4f}")

    def test_forward_forward_update(self):
        """Forward-Forward学習ステップの動作検証"""
        x_pos = torch.randn(self.batch_size, self.input_dim).to(self.device)
        x_neg = torch.randn(self.batch_size, self.input_dim).to(self.device)
        optimizer = torch.optim.Adam(self.engine.rlm.parameters(), lr=0.01)
        
        # 学習前の重みを記録
        old_weight = self.engine.rlm.W_recur.weight.data.clone()
        
        # 学習ステップ実行
        loss = self.engine.train_ff_step(x_pos, x_neg, optimizer)
        
        # Lossが正常に返されるか
        self.assertIsInstance(loss, float)
        self.assertFalse(import_math.isnan(loss), "Loss should not be NaN")
        
        # 重みが更新されているか確認
        new_weight = self.engine.rlm.W_recur.weight.data
        self.assertFalse(torch.equal(old_weight, new_weight), "Weights should be updated after training step")

import math as import_math

if __name__ == "__main__":
    unittest.main()