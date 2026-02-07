# snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampus v2 (Reinforcement Learning Enabled)
# Description: 
#   記憶構造に 'confidence' (信頼度) を追加。
#   - encode_episode: 初期信頼度 1.0 で保存。
#   - update_last_memory: 報酬信号に基づいて信頼度を増減させる。
#   - recall: 信頼度を加味して、検索スコアを調整する。

import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

class Hippocampus:
    def __init__(self, brain=None, storage_file="dora_memory_bank.json", capacity=200, input_dim=128, device='cpu'):
        self.logger = logging.getLogger("Hippocampus")
        self.storage_path = Path(storage_file)
        
        self.brain = brain
        self.device = brain.device if brain else device
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.memories = self._load_memories()
        self.memory_embeddings = self._precompute_embeddings()
        
        # 最後にアクセス/作成した記憶のインデックス
        self.last_accessed_index = -1
        
        self.logger.info(f"🧠 Hippocampus initialized. Loaded {len(self.memories)} memories.")

    def _load_memories(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    mems = json.load(f)
                    # 互換性: 古い記憶にconfidenceがない場合は1.0を追加
                    for m in mems:
                        if 'confidence' not in m:
                            m['confidence'] = 1.0
                    return mems
            except Exception:
                return []
        return []

    def _save_memories(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def _precompute_embeddings(self):
        if not self.memories:
            return None
        texts = [m['trigger'] for m in self.memories]
        return self.model.encode(texts, convert_to_tensor=True, device='cpu')

    def encode_episode(self, trigger_text, action, intensity):
        # 閾値チェック
        if intensity < 15.0:
            return None

        # 重複チェック (同じトリガーなら更新だけする)
        for i, m in enumerate(self.memories):
            if m['trigger'] == trigger_text:
                # 既存記憶を強化
                self.last_accessed_index = i
                return m

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory = {
            "timestamp": timestamp,
            "trigger": trigger_text,
            "action": action,
            "intensity": intensity,
            "confidence": 1.0 # 初期信頼度
        }
        
        self.memories.append(memory)
        self.last_accessed_index = len(self.memories) - 1
        self._save_memories()
        
        # Update Embeddings
        new_emb = self.model.encode(trigger_text, convert_to_tensor=True, device='cpu')
        if self.memory_embeddings is None:
            self.memory_embeddings = new_emb.unsqueeze(0)
        else:
            self.memory_embeddings = torch.cat([self.memory_embeddings, new_emb.unsqueeze(0)])
            
        print(f"   💾 [Hippocampus] New Memory Formed: '{trigger_text}' (Conf: 1.0)")
        return memory

    def recall(self, current_input):
        if self.memory_embeddings is None:
            return None

        query_emb = self.model.encode(current_input, convert_to_tensor=True, device='cpu')
        scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
        
        best_score_idx = torch.argmax(scores).item()
        best_score = scores[best_score_idx].item()
        
        if best_score > 0.5:
            memory = self.memories[best_score_idx]
            self.last_accessed_index = best_score_idx
            
            # Confidenceによるブースト効果
            # 信頼度が高いほど、想起時のインパクトが強い
            boosted_score = best_score * memory['confidence']
            
            print(f"   ⚡ [Hippocampus] Flashback: '{memory['trigger']}' (Conf: {memory['confidence']:.2f})")
            return memory
            
        return None

    def update_last_memory(self, reward_value):
        """
        報酬系からのフィードバックを反映する。
        reward_value: +1.0 (Good) or -1.0 (Bad)
        """
        if self.last_accessed_index == -1 or self.last_accessed_index >= len(self.memories):
            return "No recent memory to update."

        memory = self.memories[self.last_accessed_index]
        old_conf = memory['confidence']
        
        # 学習率 0.2
        new_conf = old_conf + (reward_value * 0.2)
        
        # 範囲制限 (0.1 ~ 5.0)
        new_conf = max(0.1, min(new_conf, 5.0))
        memory['confidence'] = new_conf
        
        self.memories[self.last_accessed_index] = memory
        self._save_memories()
        
        effect = "STRENGTHENED" if reward_value > 0 else "WEAKENED"
        print(f"   🧠 [Plasticity] Memory '{memory['trigger']}' {effect}. (Conf: {old_conf:.1f} -> {new_conf:.1f})")
        return new_conf