# directory: tests/models/experimental
# file: test_sara_engine.py
# purpose: Unit tests for SARA Engine v10.0
# description: SARAEngine v10.0 (Active Inference & Predictive Coding integrated) の仕様に合わせてテストを修正。

import pytest
import torch
import torch.nn as nn
from snn_research.models.experimental.sara_engine import SARAEngine, SARAMemory

# テスト用の簡易Configクラス
class SARAConfig:
    def __init__(self, hidden_size, input_size, **kwargs):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.physics = {}
        self.__dict__.update(kwargs)

class TestSARAEngine:
    
    @pytest.fixture
    def config(self):
        return SARAConfig(
            hidden_size=64, 
            input_size=32, 
            plasticity_mode='surprise_modulated', 
            reasoning_depth=2, 
            use_world_model=True, 
            learning_rate=0.01, 
            trace_decay=0.95, 
            world_model_hidden_dim=256, 
            surprise_scale=0.1
        )

    @pytest.fixture
    def engine(self, config):
        """SARAEngineのインスタンスを作成"""
        action_dim = 4 # テスト用にアクション次元を定義
        
        # SARAEngine v10.0 のシグネチャに合わせて修正
        return SARAEngine(
            input_dim=config.input_size,
            hidden_dim=config.hidden_size,
            action_dim=action_dim,
            config=config.__dict__
        )

    def test_initialization(self, engine):
        assert isinstance(engine, SARAEngine)
        assert engine.input_dim == 32
        assert engine.hidden_dim == 64
        assert engine.action_dim == 4
        # コンポーネントの存在確認
        assert hasattr(engine, 'perception_core')
        assert hasattr(engine, 'action_generator')
        assert hasattr(engine, 'memory')

    def test_forward_pass(self, engine):
        batch_size = 4
        # (Batch, Input)
        sensory_input = torch.randn(batch_size, engine.input_dim)
        # (Batch, Action) - 前回の行動
        prev_action = torch.zeros(batch_size, engine.action_dim)
        
        # 初期状態
        prev_state = engine.get_initial_state(batch_size, torch.device("cpu"))
        
        # Forward実行
        output = engine(sensory_input, prev_action, prev_state)
        
        # 出力キーの確認
        assert "action" in output
        assert "next_state" in output
        assert "pred_sensory" in output
        assert "loss_components" in output
        
        # シェイプ確認
        assert output["action"].shape == (batch_size, engine.action_dim)
        assert output["pred_sensory"].shape == (batch_size, engine.input_dim)

    def test_adaptation_step(self, engine):
        """可塑性とエネルギーの更新確認"""
        batch_size = 2
        sensory = torch.randn(batch_size, engine.input_dim)
        prev_action = torch.randn(batch_size, engine.action_dim)
        state = engine.get_initial_state(batch_size, torch.device("cpu"))
        
        # 実行前のパラメータ状態
        initial_plasticity = engine.plasticity_level.item()
        
        # 大きな誤差を生む入力で実行
        output = engine(sensory, prev_action, state)
        
        # 可塑性が変化しているか (メタ認知による調整)
        assert engine.plasticity_level.item() != initial_plasticity or engine.energy_reserve.item() < 1.0

    def test_memory_continuity(self, engine):
        """メモリと状態の連続性テスト"""
        batch_size = 1
        sensory = torch.randn(batch_size, engine.input_dim)
        prev_action = torch.zeros(batch_size, engine.action_dim)
        state = engine.get_initial_state(batch_size, torch.device("cpu"))
        
        # Step 1
        out1 = engine(sensory, prev_action, state)
        state2 = out1["next_state"]
        
        # Step 2
        out2 = engine(sensory, out1["action"], state2)
        
        assert out2["action"].shape == (batch_size, engine.action_dim)

    def test_batch_processing(self, engine):
        batch_size = 8
        sensory = torch.randn(batch_size, engine.input_dim)
        prev_action = torch.zeros(batch_size, engine.action_dim)
        state = engine.get_initial_state(batch_size, torch.device("cpu"))
        
        output = engine(sensory, prev_action, state)
        assert output["action"].shape[0] == batch_size