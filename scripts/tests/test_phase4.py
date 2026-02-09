# directory: scripts/tests
# file: test_phase4.py
# purpose: Test Phase 4 Visual Agent components
# description: 視覚エージェントの各コンポーネントの連携テスト。
#              クラス名変更 (TestTimeAdaptationWrapper -> TimeAdaptationWrapper) に対応。

import pytest
import torch
import torch.nn as nn
from snn_research.models.visual_cortex import VisualCortex
from snn_research.adaptive.active_inference_agent import ActiveInferenceAgent
# 修正: 正しいクラス名をインポート
from snn_research.adaptive.test_time_adaptation import TimeAdaptationWrapper

def test_visual_active_inference_loop():
    # Mock visual model
    visual_model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32) # State dim
    )
    
    # Initialize Agent
    agent = ActiveInferenceAgent(visual_model, state_dim=32, action_dim=5)
    
    # Simulate input
    visual_input = torch.randn(1, 128)
    
    # 1. Perception & Action Generation
    action = agent.infer_action(visual_input)
    assert action.shape == (1, 5)
    
    # 2. Adaptation (Test-time learning)
    # 修正: クラス名の変更を反映
    adapter = TimeAdaptationWrapper(visual_model)
    loss = adapter.adapt(visual_input, lambda x: x.pow(2).mean())
    assert loss is not None