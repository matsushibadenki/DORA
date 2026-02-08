# directory: tests/models/experimental
# filename: test_sara_engine.py
# description: SARAエンジンのユニットテスト (v7.4 Corrected Imports & Features)

import pytest
import torch
import sys
import os

# プロジェクトルートへのパス解決
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# 正しいパスからのインポート
from snn_research.models.experimental.sara_engine import (
    SARABrainCore,
    SARAEngine,
    SNNEncoder,
    LegendreSpikeAttractor,
    RecursiveMeaningLayer,
    RecursionController
)

class TestSARAEngine:
    
    @pytest.fixture
    def config(self):
        return {
            "input_dim": 32,
            "hidden_dim": 64,
            "output_dim": 10,
            "use_cuda": False,
            "enable_rlm": True,
            "enable_attractor": True
        }

    def test_recursion_controller(self):
        """再帰コントローラの動作テスト"""
        dim = 16
        controller = RecursionController(dim)
        x = torch.randn(4, dim)
        state = torch.randn(4, dim)
        
        out = controller(x, state)
        assert out.shape == (4, dim)
        # 出力が計算されているか（入力と異なるか）
        assert not torch.equal(out, x)

    def test_recursive_meaning_layer(self):
        """再帰層の動作テスト"""
        dim = 16
        rml = RecursiveMeaningLayer(dim)
        x = torch.randn(4, dim)
        
        out1, state1 = rml(x)
        assert out1.shape == (4, dim)
        
        x2 = torch.randn(4, dim)
        out2, state2 = rml(x2, prev_state=state1)
        assert not torch.equal(state1, state2)

    def test_encoder_and_core(self, config):
        """統合テスト"""
        model = SARABrainCore(**config)
        batch_size = 2
        x = torch.rand(batch_size, config["input_dim"])
        
        output, state = model(x)
        assert output.shape == (batch_size, config["output_dim"])
        
    def test_synaptic_plasticity(self, config):
        """RLM可塑性のテスト"""
        model = SARABrainCore(**config)
        x = torch.rand(2, config["input_dim"])
        
        out, _ = model(x)
        loss = out.sum()
        loss.backward()
        
        w_before = model.readout.weight.data.clone()
        model.apply_reward(1.0)
        model.update_synapses()
        w_after = model.readout.weight.data
        
        assert not torch.equal(w_before, w_after)

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))