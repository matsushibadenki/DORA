# directory: snn_research/models/adapters
# filename: sara_adapter.py
# description: Legacyなモデルインターフェースを最新のSARAエンジン(Rust backend)へ適合させるアダプター

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

# SARAエンジンのインポート (パスはプロジェクト構造に合わせて調整)
try:
    from snn_research.models.experimental.sara_engine import SARABrainCore
except ImportError:
    # 循環参照やRustビルド未完了時のフォールバック用ダミー
    print("Warning: SARABrainCore could not be imported. Using Mock.")
    SARABrainCore = None 

class SaraAdapter(nn.Module):
    """
    SARA (Spiking Attractor Recursive Architecture) エンジンを
    既存のPyTorchモデルインターフェースとして振る舞わせるためのアダプタークラス。
    
    主な機能:
    - 入力テンソルのRustカーネルへの受け渡し
    - 学習モード(Online/Offline)の切り替え
    - 報酬信号のRLM(Reinforcement Learning Mechanism)への伝達
    - 内部状態(短期記憶/長期記憶)の管理
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.input_size = config.get("input_size", 784)
        self.hidden_size = config.get("hidden_size", 512)
        self.output_size = config.get("output_size", 10)
        self.use_rust = config.get("use_rust_kernel", True)
        
        # SARAコアエンジンの初期化
        if SARABrainCore is not None:
            self.engine = SARABrainCore(
                input_dim=self.input_size,
                hidden_dim=self.hidden_size,
                output_dim=self.output_size,
                use_cuda=torch.cuda.is_available(),
                enable_rlm=config.get("enable_rlm", True),
                enable_attractor=config.get("enable_attractor", True)
            )
        else:
            raise ImportError("SARABrainCore is required but not found.")

        # ダミーのパラメータ登録 (Optimizerが空のパラメータリストでエラーになるのを防ぐため)
        # 実際の学習はRustカーネル内のシナプス荷重更新で行われるため、これは勾配を持たない
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        順伝播処理。
        Args:
            x: Input tensor [Batch, Time, Channels] or [Batch, Channels]
        Returns:
            Output spike probabilities or membrane potentials
        """
        # SARAエンジンへの委譲
        # Rust側が [Batch, Dim] を期待している場合、Time次元ループはここで処理するかRustに任せる
        # ここではSARAがシーケンス全体を一括処理できると仮定
        
        output, state = self.engine.forward(x)
        
        # 戻り値がTupleならTensorのみを返す（既存互換性）
        if isinstance(output, tuple):
            return output[0]
        return output

    def learn(self, x: torch.Tensor, reward: float = 0.0, target: Optional[torch.Tensor] = None):
        """
        オンライン学習実行用メソッド。
        従来のSTDP学習や強化学習のステップをこの1メソッドで完結させる。
        """
        if self.training:
            # 報酬信号をエンジンへ注入 (RLM: Reward Modulated STDP)
            self.engine.apply_reward(reward)
            
            # 教師あり学習信号がある場合 (Supervised STDP / Error-driven)
            if target is not None:
                self.engine.apply_error_signal(target)
            
            # 重み更新のトリガー
            self.engine.update_synapses()

    def get_memory_state(self) -> Dict[str, torch.Tensor]:
        """
        現在の記憶状態（短期・長期）を取得する。
        可視化やデバッグに使用。
        """
        return {
            "short_term": self.engine.get_stm_state(),
            "long_term": self.engine.get_ltm_weights(),
            "attractor": self.engine.get_attractor_energy()
        }

    def consolidate_memory(self):
        """
        睡眠フェーズなどで呼び出し。
        短期記憶を長期記憶へ転送し、構造的可塑性を適用する。
        """
        self.engine.consolidate()
        print("SARA: Memory consolidated and structural plasticity applied.")