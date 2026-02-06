# ファイルパス: snn_research/social/theory_of_mind.py
# 日本語タイトル: Theory of Mind (ToM) Module v1.6 (Encoder Mode)
# 修正内容: BitSpikeMambaをエンコーダモード（output_head=False）で初期化し、次元不一致を解消。

import torch
import torch.nn as nn
import logging
from typing import Optional, Type, TYPE_CHECKING, Any, Dict, Deque
from collections import deque, defaultdict

if TYPE_CHECKING:
    pass

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    BitSpikeMamba = None  # type: ignore

logger = logging.getLogger(__name__)


class TheoryOfMindModule(nn.Module):
    """
    心の理論モジュール。
    他者の行動を観測し、意図や信頼度を推定する。
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        observation_dim: Optional[int] = None,
        hidden_dim: int = 64,
        intent_dim: int = 8,
        model_type: str = "mamba",
        history_len: int = 16
    ):
        super().__init__()

        # input_dimの解決（observation_dimも許容）
        self.input_dim = input_dim if input_dim is not None else (
            observation_dim if observation_dim is not None else 4)
        self.model_type = model_type
        self.history_len = history_len

        # 履歴管理用のバッファ
        self.interaction_history: Dict[str, Deque[torch.Tensor]] = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )

        self.core: nn.Module
        self.input_proj: nn.Module

        if model_type == "mamba" and BitSpikeMamba is not None:
            MambaClass: Type[nn.Module] = BitSpikeMamba  # type: ignore
            self.core = MambaClass(
                vocab_size=1, # ダミー（Embeddingは使われない）
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
                num_layers=2,
                time_steps=history_len,
                neuron_config={"type": "lif"},
                output_head=False # [Fix] エンコーダとして使うためヘッドを無効化
            )
            self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        else:
            if model_type == "mamba":
                logger.warning("BitSpikeMamba not found. Falling back to GRU.")
            self.core = nn.GRU(
                input_size=self.input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True
            )
            self.input_proj = nn.Identity()

        self.intent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, intent_dim)
        )

    def forward(self, observation_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation_sequence: [Batch, Time, Dim]
        """
        x = self.input_proj(observation_sequence)

        if isinstance(self.core, nn.GRU):
            out, _ = self.core(x)
            final_state = out[:, -1, :]
        else:
            # Mamba core returns [Batch, Time, Hidden] when output_head=False
            mamba_out = self.core(x)
            features = mamba_out[0] if isinstance(mamba_out, tuple) else mamba_out
            
            # 最終時刻の状態を取得 [Batch, Hidden]
            if features.dim() == 3:
                final_state = features[:, -1, :]
            else:
                final_state = features

        return self.intent_head(final_state)

    def predict_goal(self, trajectory: torch.Tensor) -> torch.Tensor:
        """ゴール予測 (推論用)"""
        self.eval()
        with torch.no_grad():
            return self.forward(trajectory)

    def predict_action(self, agent_id: str) -> torch.Tensor:
        return torch.rand(1, requires_grad=False)

    def update_model(self, target_id: str, outcome: Any):
        pass

    def observe_agent(self, agent_id: str, action: torch.Tensor):
        self.interaction_history[agent_id].append(action)


# Alias for backward compatibility
TheoryOfMindEncoder = TheoryOfMindModule