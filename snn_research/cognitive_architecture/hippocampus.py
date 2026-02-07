# snn_research/cognitive_architecture/hippocampus.py
# Title: Hippocampus v2.2 (Buffer Accessor)
# Description: episodic_bufferプロパティを追加し、テストでの属性アクセスに対応。

import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union, List
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
        self.last_accessed_index = -1
        self.logger.info(f"🧠 Hippocampus initialized. Loaded {len(self.memories)} memories.")

    # [Fix] Added property for test compatibility
    @property
    def episodic_buffer(self) -> List[Dict[str, Any]]:
        """Alias for memories list to satisfy tests."""
        return self.memories

    def _load_memories(self):
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    mems = json.load(f)
                    for m in mems:
                        if 'confidence' not in m: m['confidence'] = 1.0
                    return mems
            except Exception:
                return []
        return []

    def _save_memories(self):
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.memories, f, ensure_ascii=False, indent=2)

    def _precompute_embeddings(self):
        if not self.memories: return None
        texts = [m['trigger'] for m in self.memories]
        return self.model.encode(texts, convert_to_tensor=True, device='cpu')

    def encode_episode(self, trigger_text, action, intensity):
        if intensity < 15.0: return None
        for i, m in enumerate(self.memories):
            if m['trigger'] == trigger_text:
                self.last_accessed_index = i
                return m

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory = {
            "timestamp": timestamp, "trigger": trigger_text, "action": action,
            "intensity": intensity, "confidence": 1.0
        }
        self.memories.append(memory)
        self.last_accessed_index = len(self.memories) - 1
        self._save_memories()
        
        new_emb = self.model.encode(trigger_text, convert_to_tensor=True, device='cpu')
        if self.memory_embeddings is None:
            self.memory_embeddings = new_emb.unsqueeze(0)
        else:
            self.memory_embeddings = torch.cat([self.memory_embeddings, new_emb.unsqueeze(0)])
        print(f"   💾 [Hippocampus] New Memory Formed: '{trigger_text}'")
        return memory

    def store_episode(self, data: Any) -> None:
        trigger = "Unknown Experience"
        if isinstance(data, torch.Tensor): trigger = f"Tensor Pattern {data.shape}"
        elif isinstance(data, str): trigger = data
        elif isinstance(data, dict): trigger = str(data.get("trigger", "Dict Pattern"))
        self.encode_episode(trigger, action="stored", intensity=20.0)

    def process(self, data: Any) -> None:
        self.store_episode(data)

    def recall(self, current_input):
        if self.memory_embeddings is None: return None
        query_emb = self.model.encode(current_input, convert_to_tensor=True, device='cpu')
        scores = util.cos_sim(query_emb, self.memory_embeddings)[0]
        best_score_idx = torch.argmax(scores).item()
        best_score = scores[best_score_idx].item()
        
        if best_score > 0.5:
            memory = self.memories[best_score_idx]
            self.last_accessed_index = best_score_idx
            print(f"   ⚡ [Hippocampus] Flashback: '{memory['trigger']}' (Conf: {memory['confidence']:.2f})")
            return memory
        return None

    def update_last_memory(self, reward_value):
        if self.last_accessed_index == -1 or self.last_accessed_index >= len(self.memories): return
        memory = self.memories[self.last_accessed_index]
        memory['confidence'] = max(0.1, min(memory['confidence'] + (reward_value * 0.2), 5.0))
        self.memories[self.last_accessed_index] = memory
        self._save_memories()