# directory: tests
# file: test_active_inference.py
# purpose: Test Active Inference Agent
# description: ActiveInferenceAgentの基本動作テスト。
#              クラス名変更 (TestTimeAdaptationWrapper -> TimeAdaptationWrapper) に対応。

import pytest
import torch
import torch.nn as nn
from snn_research.adaptive.active_inference_agent import ActiveInferenceAgent
# 修正: 正しいクラス名をインポート
from snn_research.adaptive.test_time_adaptation import TimeAdaptationWrapper

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.layer(x)

def test_active_inference_initialization():
    model = MockModel()
    agent = ActiveInferenceAgent(model, state_dim=10, action_dim=4)
    assert agent is not None
    assert agent.model is model

def test_inference_step():
    model = MockModel()
    agent = ActiveInferenceAgent(model, state_dim=10, action_dim=4)
    
    obs = torch.randn(1, 10)
    # goal_state も必要であれば設定
    action = agent.infer_action(obs)
    
    assert action.shape == (1, 4)

def test_wrapper_adaptation():
    model = MockModel()
    # 修正: クラス名の変更を反映
    wrapper = TimeAdaptationWrapper(model)
    
    x = torch.randn(5, 10)
    output = wrapper(x)
    assert output.shape == (5, 10)
    
    # 簡易的なロス関数
    loss_fn = lambda y: y.mean()
    loss = wrapper.adapt(x, loss_fn)
    assert loss is not None