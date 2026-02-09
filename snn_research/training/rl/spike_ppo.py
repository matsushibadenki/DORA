# directory: snn_research/training/rl
# file: spike_ppo.py
# title: Spike PPO (SARA Integrated & Bug Fixed)
# purpose: PPOアルゴリズムのスパイキング実装。SARA Engine v7.4をバックエンドとして統合。
#          LegacyモードでのNameError(F)およびNormal分布の次元エラーを修正済み。

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
from typing import Tuple, Any, Optional, Union

# SARA Engine のインポート試行
try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    SARAEngine = None

class SARAActorCritic(nn.Module):
    """SARA Engine を利用した Actor-Critic ネットワーク"""
    def __init__(self, input_dim: int, action_dim: int, continuous_action: bool):
        super().__init__()
        self.continuous_action = continuous_action
        
        # SARA Engine の初期化
        self.engine = SARAEngine(
            input_dim=input_dim,
            version="v7.4",
            perception_mode="active_inference"
        )
        
        # SARAの隠れ層サイズを取得（デフォルト256と仮定）
        hidden_dim = getattr(self.engine, 'hidden_dim', 256)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)
        
        if continuous_action:
            # 標準偏差を保持（1次元ベクトルとして扱う）
            self.action_std = nn.Parameter(torch.ones(action_dim) * 0.5)

    def act(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """行動のサンプリング"""
        h = self.engine(state, output_mode="hidden")
        action_logits = self.actor_head(h)
        state_value = self.critic_head(h)

        if self.continuous_action:
            # 連続値: 正規分布 (scaleは標準偏差ベクトル)
            dist = Normal(action_logits, F.softplus(self.action_std))
        else:
            # 離散値: カテゴリカル分布
            dist = Categorical(F.softmax(action_logits, dim=-1))

        action = dist.sample()
        return action.detach(), dist.log_prob(action).detach(), state_value.detach()

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """蓄積されたデータの評価"""
        h = self.engine(state, output_mode="hidden")
        action_logits = self.actor_head(h)
        state_value = self.critic_head(h)

        if self.continuous_action:
            dist = Normal(action_logits, F.softplus(self.action_std))
        else:
            dist = Categorical(F.softmax(action_logits, dim=-1))

        return dist.log_prob(action), state_value, dist.entropy()

class SpikePPO:
    """PPOエージェント。SARAまたはLegacyバックエンドを選択可能。"""
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous_action: bool = False,
        use_sara_backend: bool = True,
        **kwargs
    ):
        self.device = torch.device("cpu")
        
        if use_sara_backend and SARAEngine:
            print("SpikePPO: Initializing with SARA Engine v7.4 Backend.")
            self.policy = SARAActorCritic(state_dim, action_dim, continuous_action).to(self.device)
        else:
            print("SpikePPO: Initializing with Legacy AC Backend.")
            
            # 内部クラスとしてLegacyACを定義（Fの定義不足を解消）
            class LegacyAC(nn.Module):
                def __init__(self, s, a, c):
                    super().__init__()
                    self.cont = c
                    self.net = nn.Sequential(nn.Linear(s, 128), nn.Tanh(), nn.Linear(128, a))
                    self.val = nn.Sequential(nn.Linear(s, 128), nn.Tanh(), nn.Linear(128, 1))
                    if c: self.std = nn.Parameter(torch.ones(a) * 0.5)
                def act(self, x):
                    mu = self.net(x)
                    v = self.val(x)
                    # 修正: F.softplus/softmaxを使用
                    dist = Normal(mu, F.softplus(self.std)) if self.cont else Categorical(F.softmax(mu, -1))
                    act = dist.sample()
                    return act, dist.log_prob(act), v
                def evaluate(self, x, a):
                    mu = self.net(x)
                    v = self.val(x)
                    dist = Normal(mu, F.softplus(self.std)) if self.cont else Categorical(F.softmax(mu, -1))
                    return dist.log_prob(a), v, dist.entropy()
            
            self.policy = LegacyAC(state_dim, action_dim, continuous_action).to(self.device)
            
        self.policy_old = self.policy
        # パラメータ全体を最適化
        self.optimizer = optim.Adam(self.policy.parameters(), lr=kwargs.get('lr', 1e-3))
        self.buffer = []

    def select_action(self, state: Union[torch.Tensor, Any]) -> Any:
        """エピソード中のアクション選択"""
        with torch.no_grad():
            s = state if isinstance(state, torch.Tensor) else torch.FloatTensor(state)
            if s.dim() == 1: s = s.unsqueeze(0)
            a, lp, v = self.policy_old.act(s)
        
        self.buffer.append({
            'state': s, 'action': a, 'logprob': lp, 
            'value': v, 'reward': 0, 'done': False
        })
        return a.item() if a.numel() == 1 else a.cpu().numpy().flatten()

    def store_reward(self, reward: float, done: bool):
        """報酬の記録"""
        if self.buffer:
            self.buffer[-1]['reward'] = reward
            self.buffer[-1]['done'] = done

    def update(self):
        """ポリシーの更新（バッファのクリアのみ、具体的な学習ロジックは必要に応じて呼び出す）"""
        self.buffer = []