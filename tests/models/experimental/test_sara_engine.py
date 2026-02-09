# directory: tests/models/experimental
# file: test_sara_engine.py
# purpose: Unit tests for SARA Engine v9.0
# description: SARAEngine v9.0 (Active Inference & Predictive Coding integrated) の仕様に合わせてテストを修正。
#              戻り値のアンパック数(3つ)や属性チェック(surprise_detector削除)を更新。

import pytest
import torch
import sys
import os

# プロジェクトルートへのパス解決
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from snn_research.models.experimental.sara_engine import SARAEngine, SARAMemory
from snn_research.config.schema import SARAConfig

class TestSARAEngine:
    @pytest.fixture
    def config(self):
        """テスト用の設定を作成"""
        return SARAConfig(
            hidden_size=64,
            input_size=32,
            plasticity_mode="surprise_modulated",
            reasoning_depth=2,
            use_world_model=True
        )

    @pytest.fixture
    def engine(self, config):
        """SARAEngineのインスタンスを作成"""
        return SARAEngine(config)

    def test_initialization(self, engine):
        """モデルが正しく初期化されるかテスト"""
        assert isinstance(engine, SARAEngine)
        assert engine.world_model is not None
        
        # v9.0: surprise_detector は廃止(None)され、Predictive Coding Unit に移行しました
        # assert engine.surprise_detector is not None 
        
        # 新しいコンポーネントの確認
        assert engine.sensory_pe_unit is not None
        assert engine.state_pe_unit is not None
        assert engine.active_inference is not None
        
        # Traceバッファの確認
        assert hasattr(engine, 'spike_trace')

    def test_forward_pass(self, engine):
        """順伝播の入出力形状確認"""
        batch_size = 1
        inputs = torch.randn(batch_size, 32)
        
        # v9.0修正: 戻り値は (action, memory, info) の3つ
        action, memory, info = engine(inputs)
        
        # 行動出力チェック (デフォルトでinput_sizeと同じ次元)
        assert action.shape == (batch_size, 32)
        
        # メモリ状態チェック
        assert isinstance(memory, SARAMemory)
        assert memory.hidden_state.shape == (batch_size, 64)
        assert memory.synaptic_trace.shape == (batch_size, 64)
        
        # infoの内容チェック
        assert "free_energy" in info
        assert "energy" in info

    def test_adaptation_step(self, engine):
        """学習（適応）ステップの動作確認 (adaptメソッド)"""
        inputs = torch.randn(1, 32)
        
        # 実行 (adapt: 推論 + 誤差計算 + 重み更新)
        results = engine.adapt(inputs)
        
        # 結果のキー確認
        assert "loss" in results
        assert "surprise" in results
        assert "memory" in results
        assert "outputs" in results # v9.0では 'outputs' は action を指す
        assert "info" in results
        
        # Surprise (Free Energy) が計算されているか
        surprise = results["surprise"]
        # shapeは (Batch, 1) または (Batch, 1) スカラー相当
        assert surprise.numel() == 1 or surprise.shape[0] == 1
        assert isinstance(surprise, torch.Tensor)

    def test_memory_continuity(self, engine):
        """記憶（隠れ状態）が次のステップに引き継がれるか"""
        inputs1 = torch.randn(1, 32)
        # v9.0修正: 3つアンパック
        _, mem1, _ = engine(inputs1)
        
        inputs2 = torch.randn(1, 32)
        # 前のメモリを渡して次のステップを実行
        # v9.0修正: 3つアンパック
        _, mem2, _ = engine(inputs2, prev_memory=mem1)
        
        # 別のオブジェクトになっているはず
        assert mem2 is not mem1
        # World Modelの状態が更新されていること
        assert mem2.world_model_state is not None

    def test_batch_processing(self, engine):
        """バッチ処理の確認"""
        batch_size = 4
        inputs = torch.randn(batch_size, 32)
        # v9.0修正: 3つアンパック
        outputs, _, _ = engine(inputs)
        assert outputs.shape == (batch_size, 32)

if __name__ == "__main__":
    pytest.main([__file__])