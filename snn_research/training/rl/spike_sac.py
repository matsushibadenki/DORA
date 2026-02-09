# directory: snn_research/training/rl
# file: spike_sac.py
# title: Spike SAC (Soft Actor-Critic) with SARA Integrated
# purpose: SARA Engine v7.4を特徴抽出のコアとしたSACアルゴリズムの実装。
#          連続値制御タスクにおける探索性能と安定性を最大化する。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Any, Dict

try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    SARAEngine = None

class SARASACPolicy(nn.Module):
    """SARAをバックボーンとするSACポリシー＆ダブルQ関数"""
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # 共有特徴抽出器としてのSARA
        self.engine = SARAEngine(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            version="v7.4",
            perception_mode="active_inference"
        )
        
        # Actor: 平均と対数標準偏差
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Critic: 2つの独立したQネットワーク
        self.critic1 = nn.Sequential(nn.Linear(hidden_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))
        self.critic2 = nn.Sequential(nn.Linear(hidden_dim + action_dim, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.engine(state, output_mode="hidden")
        return self.actor_mean(h), torch.clamp(self.actor_log_std(h), -20, 2)

    def sample_action(self, state: torch.Tensor):
        """再パラメータ化トリックを用いたサンプリング"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t) # 行動を [-1, 1] に制限
        
        # ヤコビアン補正項を含む対数確率
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        return y_t, log_prob.sum(1, keepdim=True), torch.tanh(mean)

    def get_q_values(self, state: torch.Tensor, action: torch.Tensor):
        h = self.engine(state, output_mode="hidden")
        sa = torch.cat([h, action], dim=1)
        return self.critic1(sa), self.critic2(sa)

class SpikeSAC:
    def __init__(self, state_dim: int, action_dim: int, use_sara_backend: bool = True, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = kwargs.get("gamma", 0.99)
        self.tau = kwargs.get("tau", 0.005)
        self.alpha = kwargs.get("alpha", 0.2)
        
        if use_sara_backend and SARAEngine:
            print("SpikeSAC: Integrated with SARA Engine v7.4")
            self.policy = SARASACPolicy(state_dim, action_dim).to(self.device)
            self.target_policy = SARASACPolicy(state_dim, action_dim).to(self.device)
            self.target_policy.load_state_dict(self.policy.state_dict())
        else:
            # Legacy implementation omitted for brevity
            pass
            
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

    def select_action(self, state: np.ndarray, evaluate: bool = False):
        state_t = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, _, mean = self.policy.sample_action(state_t)
        return mean.cpu().numpy()[0] if evaluate else action.cpu().numpy()[0]