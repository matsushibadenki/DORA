# directory: scripts/experiments/brain
# file: run_lifelong_learning_test.py
# purpose: Test script for lifelong learning capabilities
# description: 継続学習の能力をテストするスクリプト。
#              不足していた 'brain' フィクスチャを追加し、SARA Engineを用いたテストを実行可能にします。

import pytest
import torch
import sys
import os

# プロジェクトルートへのパス解決
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from snn_research.models.experimental.sara_engine import SARAEngine
from snn_research.config.schema import SARAConfig

# フィクスチャの追加
@pytest.fixture
def brain():
    config = SARAConfig(
        hidden_size=64,
        input_size=32,
        plasticity_mode="surprise_modulated",
        use_world_model=True
    )
    return SARAEngine(config)

@pytest.fixture
def inputs():
    return torch.randn(1, 32)

@pytest.fixture
def label():
    return torch.randn(1, 32) # 自己教師あり的なターゲット

def test_brain(brain, inputs, label):
    """
    SARA Engineが入力とターゲットを受け取り、学習ステップを実行できるか確認
    """
    results = brain.adapt(inputs, targets=label)
    
    assert "loss" in results
    assert "surprise" in results
    assert results["loss"].item() >= 0