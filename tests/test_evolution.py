# ファイルパス: tests/test_evolution.py
# 日本語タイトル: Self-Evolving Agent Unit Tests (Syntax Fixed)
# 目的: 自己進化エージェントの学習パラメータおよびニューロンタイプの進化ロジックをテストする。
# 修正内容: 23行目のファイルオープン処理の構文エラー（with open("の欠落）を修正。

import pytest
import os
import shutil
from unittest.mock import MagicMock
from snn_research.agent.self_evolving_agent import SelfEvolvingAgentMaster
from snn_research.cognitive_architecture.hierarchical_planner import HierarchicalPlanner
from snn_research.distillation.model_registry import ModelRegistry
from app.services.web_crawler import WebCrawler
from snn_research.cognitive_architecture.meta_cognitive_snn import MetaCognitiveSNN
from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem

# Mock dependencies


@pytest.fixture
def mock_agent():
    # Setup paths
    os.makedirs("tests/temp_configs", exist_ok=True)
    with open("tests/temp_configs/test_training.yaml", "w") as f:
        f.write(
            "training:\n  gradient_based:\n    learning_rate: 0.001\n    loss:\n      weight_decay: 0.0001")

    # Fix: Added 'with open("' to start of line
    with open("tests/temp_configs/test_model.yaml", "w") as f:
        f.write("model:\n  d_model: 128\n  neuron:\n    type: \"lif\"")

    agent = SelfEvolvingAgentMaster(
        name="TestAgent",
        planner=MagicMock(spec=HierarchicalPlanner),
        model_registry=MagicMock(spec=ModelRegistry),
        memory=MagicMock(),
        web_crawler=MagicMock(spec=WebCrawler),
        meta_cognitive_snn=MagicMock(spec=MetaCognitiveSNN),
        motivation_system=MagicMock(spec=IntrinsicMotivationSystem),
        training_config_path="tests/temp_configs/test_training.yaml",
        model_config_path="tests/temp_configs/test_model.yaml"
    )
    # Redirect evolved configs output
    agent.evolved_config_dir = "tests/temp_configs/evolved"
    os.makedirs(agent.evolved_config_dir, exist_ok=True)

    yield agent

    # Cleanup
    shutil.rmtree("tests/temp_configs")


def test_evolve_learning_parameters(mock_agent):
    result = mock_agent._evolve_learning_parameters(
        performance_eval={}, internal_state={}, scope="global"
    )
    assert "Successfully evolved parameters" in result
    assert "test_training_params_" in result


def test_evolve_neuron_type(mock_agent):
    result = mock_agent._evolve_neuron_type(
        performance_eval={}, internal_state={}
    )
    assert "Successfully evolved neuron type" in result
    assert "test_model_neuron_" in result