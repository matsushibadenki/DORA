# ファイルパス: app/deployment.py
# 日本語タイトル: SNN Inference Engine v2.0 (Associative Memory)
# 目的・内容:
#   Neuromorphic OSのダイナミクスを活用した高度な推論エンジン。
#   入力に対する「思考ループ（反芻）」と、スパイク類似度に基づく「連想記憶検索」を行い、
#   文脈を踏まえた応答を生成する。

import logging
import torch
import torch.nn.functional as F
import random
import time
from typing import Dict, Any, Optional, List, Tuple

from snn_research.core.neuromorphic_os import NeuromorphicOS
from snn_research.io.spike_encoder import TextSpikeEncoder
from snn_research.io.spike_decoder import RateDecoder

logger = logging.getLogger(__name__)

class SNNInferenceEngine:
    """
    Neuromorphic OSを実行するためのランタイムエンジン。
    短期記憶(Working Memory)と連想想起(Associative Recall)を活用する。
    """
    
    def __init__(self, brain: NeuromorphicOS, config: Dict[str, Any]):
        self.brain = brain
        self.config = config
        self.device = self.brain.device
        
        # --- IO Systems ---
        input_dim = self.config.get("model", {}).get("input_dim", 784)
        
        # Encoder: テキスト -> スパイク (意味ベクトル)
        self.encoder = TextSpikeEncoder(
            num_neurons=input_dim, 
            device=str(self.device)
        )
        
        # Decoder: スパイク -> 抽象値
        self.decoder = RateDecoder(output_dim=input_dim, device=str(self.device))
        
        # --- Episodic Memory Store (Engine Level) ---
        # 脳(Hippocampus)は純粋なTensorのみを保持するため、
        # ここで「Tensor ⇔ テキスト意味」の対応関係を保持し、言語化をサポートする。
        # 構造: List of {'text': str, 'tensor': torch.Tensor, 'cycle': int}
        self.episodic_memory: List[Dict[str, Any]] = []
        
        logger.info(f"🤖 Context-Aware Engine ready on {self.device}")

    def generate_response(self, text: str) -> str:
        """
        テキスト入力を脳への刺激に変換し、思考ループと記憶検索を経て応答する。
        """
        # 1. Encoding (Text -> Semantic Spikes)
        # 入力テキストをスパイクパターン（意味表現）に変換
        spike_sequence = self.encoder(text, duration=10)
        sensory_input = spike_sequence.mean(dim=1) * 2.0 # 強度調整
        
        # メモリへの登録 (現在の文脈を保存)
        # 本来は睡眠時の定着を経るが、短期記憶(Working Memory)として即時利用可能とする
        current_memory = {
            "text": text,
            "tensor": sensory_input.detach(), # 勾配を切って保存
            "cycle": self.brain.cycle_count
        }
        self.episodic_memory.append(current_memory)
        # メモリ容量制限 (簡易的)
        if len(self.episodic_memory) > 50:
            self.episodic_memory.pop(0)

        # 2. Thinking Loop (Cognitive Dynamics)
        # 1回の入力に対して複数サイクル回し、脳内での情報の反響と定着を促す
        thought_steps = 5
        max_consciousness = 0.0
        active_regions = set()
        
        for step in range(thought_steps):
            # 入力刺激は時間とともに減衰させる（残響効果を見るため）
            current_stimulus = sensory_input * (1.0 / (step + 1))
            
            # OSサイクル実行 (Wake Phase)
            observation = self.brain.run_cycle(current_stimulus, phase="wake")
            
            # 意識レベルのモニタリング
            c_level = observation.get("consciousness_level", 0.0)
            max_consciousness = max(max_consciousness, c_level)
            
            # 活性化した領野の記録
            for region, activity in observation.get("substrate_activity", {}).items():
                if activity > 0.01:
                    active_regions.add(region)

        # 3. Associative Recall (Memory Retrieval)
        # 現在の入力パターンと類似した過去の記憶を探す
        recalled_text = self._perform_associative_recall(sensory_input)
        
        # 4. Response Synthesis
        return self._synthesize_response(text, max_consciousness, active_regions, recalled_text)

    def _perform_associative_recall(self, current_tensor: torch.Tensor) -> Optional[str]:
        """
        現在の入力テンソルと過去の記憶テンソルの類似度を計算し、
        閾値を超えたものを「想起」として返す。
        """
        if len(self.episodic_memory) < 2:
            return None
            
        best_sim = 0.0
        best_text = None
        
        # 最新の記憶（自分自身）は除外
        past_memories = self.episodic_memory[:-1]
        
        current_flat = current_tensor.view(1, -1)
        
        for mem in reversed(past_memories):
            mem_tensor = mem["tensor"].to(self.device).view(1, -1)
            
            # コサイン類似度計算 (意味的類似性)
            sim = F.cosine_similarity(current_flat, mem_tensor).item()
            
            if sim > best_sim:
                best_sim = sim
                best_text = mem["text"]
        
        # 閾値判定 (0.6以上で「似ている」と判断)
        # ランダムなエンコーディングの場合、閾値は調整が必要だが、
        # TextSpikeEncoderが意味的近さを反映していれば機能する
        if best_sim > 0.65:
            logger.info(f"💡 Memory Recall: '{best_text}' (Similarity: {best_sim:.2f})")
            return best_text
            
        return None

    def _synthesize_response(
        self, 
        input_text: str, 
        consciousness: float, 
        regions: set, 
        recall: Optional[str]
    ) -> str:
        """
        脳の状態と想起された記憶に基づいて応答文を生成する。
        """
        # 基本応答
        if consciousness > 0.1:
            base = f"I have processed '{input_text}' with high awareness ({consciousness:.2f})."
        else:
            base = f"Signal '{input_text}' received."
            
        # 思考の深さ
        if len(regions) > 2:
            base += " Activity spread across multiple regions."
            
        # 文脈（記憶）の活用
        if recall:
            # 過去の文脈を引用
            if recall == input_text:
                context = f"\n💡 We are discussing '{recall}' again. It seems important."
            else:
                context = f"\n💡 This reminds me of our previous topic: '{recall}'."
            return base + context
        
        return base