# directory: tests
# file: test_smoke_all_paradigms.py
# purpose: Smoke tests for all training paradigms
# description: STDPのインポート名を修正し、スキップされていたテストを解消。

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# SARA Engine
from snn_research.models.experimental.sara_engine import SARAEngine

# テスト用Config
class SARAConfig:
    def __init__(self, hidden_size, input_size, **kwargs):
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.physics = {}
        self.__dict__.update(kwargs)

@pytest.fixture
def dummy_dataloader():
    # ダミーデータ: (Batch, Features)
    inputs = torch.randn(10, 128)
    targets = torch.randint(0, 2, (10,))
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2)

def test_smoke_sara_engine(dummy_dataloader):
    """
    SARA Engine (Physics/Predictive integrated) のスモークテスト
    """
    print("\n--- Testing: SARA Engine (Physics/Predictive integrated) ---")
    
    input_dim = 128
    hidden_dim = 128
    action_dim = 10
    
    config = SARAConfig(hidden_size=hidden_dim, input_size=input_dim)
    
    # v10.0 の初期化シグネチャに修正
    brain = SARAEngine(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        config=config.__dict__
    )
    
    # 1バッチ実行
    batch = next(iter(dummy_dataloader))
    inputs, _ = batch # inputs: (2, 128)
    
    batch_size = inputs.size(0)
    state = brain.get_initial_state(batch_size, inputs.device)
    prev_action = torch.zeros(batch_size, action_dim)
    
    # Forward Pass
    output = brain(inputs, prev_action, state)
    
    assert "action" in output
    assert "sensory_error" in output
    assert output["action"].shape == (batch_size, action_dim)
    print("✅ SARA Engine Smoke Test Passed")

def test_smoke_simple_snn(dummy_dataloader):
    """従来のSNNモデルの簡易テスト"""
    from snn_research.core.snn_core import SNNCore
    
    # 修正: config引数を追加
    config = {
        "neuron_type": "LIF",
        "threshold": 1.0,
        "decay": 0.5
    }
    
    # 簡易なSNNコアのテスト
    model = SNNCore(input_size=128, hidden_size=64, output_size=2, config=config)
    
    inputs, _ = next(iter(dummy_dataloader))
    # inputs: (Batch, Input) -> (2, 128)
    
    output = model(inputs)
    
    # 辞書かTensorが返る想定
    if isinstance(output, dict):
        assert "output" in output or "logits" in output
    else:
        # 出力形状の確認 (Batch, Output) または (Batch, Time, Output)
        if output.dim() == 3:
            # shape: (Time, Batch, Output) -> batchはdim 1
            assert output.shape[1] == 2
        else:
            # shape: (Batch, Output)
            # 形状が不一致の場合はログを出してデバッグしやすくする
            assert output.shape[0] == 2, f"Output shape mismatch: {output.shape}"

def test_smoke_stdp_placeholder():
    """STDP学習則のインポート確認"""
    try:
        # モジュール自体をインポート
        import snn_research.learning_rules.stdp as stdp_module
        
        # クラス名の揺らぎに対応 (STDP または STDPLearner)
        if hasattr(stdp_module, "STDP"):
            assert True
        elif hasattr(stdp_module, "STDPLearner"):
            assert True
        else:
            pytest.fail(f"STDP module found but class 'STDP' or 'STDPLearner' missing. Available: {dir(stdp_module)}")
            
    except ImportError as e:
        pytest.fail(f"STDP module not found: {e}")

def test_smoke_concept_augmented_placeholder():
    """Concept Augmented Trainerのインポート確認"""
    try:
        from snn_research.training.trainers.concept_augmented_trainer import ConceptAugmentedTrainer
        assert True
    except ImportError as e:
        pytest.fail(f"Concept Augmented Trainer not found: {e}")