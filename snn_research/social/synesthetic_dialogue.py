# directory: snn_research/social
# file: synesthetic_dialogue.py
# purpose: 共感覚と言語を統合した対話マネージャー (Brain v4廃止 -> SARA Engine統合版)

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional

# 修正: 廃止されたBrainV4を削除し、モデルファクトリーを使用
from snn_research.core.factories import create_model
# 直接アダプターを使う場合のフォールバック
from snn_research.models.adapters.sara_adapter import SaraAdapter

class SynestheticDialogueManager(nn.Module):
    """
    共感覚対話マネージャー:
    ユーザーからのテキスト入力を、SARAエンジンの「概念空間(Concept Space)」へマッピングし、
    色・音・感情の共感覚的な内部状態を経て応答を生成する。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # 言語モデル設定 (Embeddingなど)
        self.vocab_size = config.get("vocab_size", 10000)
        self.embed_dim = config.get("embed_dim", 256)
        
        # 単純なEmbedding層 (入力テキスト -> ベクトル)
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # --- Core Brain (SARA Engine) への置き換え ---
        # BrainV4の代わりに、最新のSARAエンジンを使用する
        brain_config = config.get("brain_config", {})
        # SARA向けに設定を強制上書き
        brain_config.update({
            "model_type": "sara_engine",  # ファクトリーでSaraAdapterを生成させる
            "input_size": self.embed_dim,
            "hidden_size": 1024,
            "output_size": self.embed_dim, # 概念ベクトルとして出力
            "enable_attractor": True
        })
        
        try:
            self.brain = create_model(brain_config)
        except Exception as e:
            print(f"Warning: Factory failed ({e}), falling back to direct SaraAdapter init.")
            self.brain = SaraAdapter(brain_config)

        # 応答生成器 (概念ベクトル -> テキスト確率)
        self.decoder = nn.Linear(self.embed_dim, self.vocab_size)

    def forward(self, input_ids: torch.Tensor, emotion_state: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        対話処理の実行
        Args:
            input_ids: [Batch, SeqLen] トークンID
            emotion_state: [Batch, EmotionDim] 外部からの感情バイアス (オプション)
        """
        # 1. テキストの埋め込み
        # [Batch, Seq, Embed]
        x = self.embedding(input_ids)
        
        # 2. 時間方向の統合 (SNNへの入力用に平均化またはシーケンス処理)
        # SARAがシーケンス対応ならそのまま渡すが、ここでは簡易的に平均化
        if x.dim() > 2:
            x_snn_input = x.mean(dim=1)
        else:
            x_snn_input = x

        # 3. 共感覚脳(SARA)による処理
        # 内部でアトラクタが働き、入力刺激に近い「記憶概念」が想起される
        concept_vector = self.brain(x_snn_input)
        
        # 4. 感情による変調 (Synesthesia Effect)
        if emotion_state is not None:
            # 感情ベクトルを概念ベクトルに加算・変調
            if emotion_state.shape[-1] == concept_vector.shape[-1]:
                concept_vector = concept_vector + emotion_state * 0.5
        
        # 5. 言語へのデコード
        logits = self.decoder(concept_vector)
        
        return {
            "logits": logits,
            "concept_state": concept_vector,
            "memory_trace": self.get_memory_status()
        }

    def reply(self, text_input: str) -> str:
        """
        簡易的な応答生成メソッド (デモ用)
        """
        # ダミー処理: 実際はTokenizerが必要だが、ここでは動作確認用のモック
        dummy_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.no_grad():
            output = self.forward(dummy_ids)
        
        # SARAの内部状態が変化したことを利用して「思考した」とみなす
        return f"[SARA] I processed '{text_input}' through my attractor dynamics."

    def get_memory_status(self):
        """脳の記憶状態を取得"""
        if hasattr(self.brain, "get_memory_state"):
            return self.brain.get_memory_state()
        return {}