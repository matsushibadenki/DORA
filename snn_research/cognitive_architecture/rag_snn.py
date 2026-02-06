# ファイルパス: snn_research/cognitive_architecture/rag_snn.py
# 日本語タイトル: Spiking RAG System v2.0 (Persistence Enabled)
# 目的: 知識ベースの永続化(save/load)を追加し、システムの再起動後も記憶を維持可能にする。

import logging
import torch
import os
import json
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, embedding_dim: Optional[int] = None, vector_store_path: Optional[str] = None):
        self.knowledge_base: List[str] = []
        self.metadata_store: List[Dict[str, Any]] = []
        self.embeddings: Optional[torch.Tensor] = None
        self.vector_store_path = vector_store_path

        # 起動時に自動ロードを試みる
        if self.vector_store_path:
            try:
                os.makedirs(self.vector_store_path, exist_ok=True)
                self.load(self.vector_store_path)
                logger.info(f"📁 Vector store initialized at: {self.vector_store_path}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize vector store: {e}")

        self.has_encoder = False
        self.encoder: Any = None

        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
            self.has_encoder = True
            detected_dim = self.encoder.get_sentence_embedding_dimension()
            self.embedding_dim = int(detected_dim) if isinstance(
                detected_dim, int) else 384
        except ImportError:
            logger.warning(
                "⚠️ sentence_transformers not found. Using random embeddings.")
            self.embedding_dim = embedding_dim if embedding_dim is not None else 768

    def _encode(self, texts: List[str]) -> torch.Tensor:
        if self.has_encoder and self.encoder:
            embeddings = self.encoder.encode(texts, convert_to_tensor=True)
            return embeddings.cpu()
        return torch.randn(len(texts), self.embedding_dim)

    def add_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        chunks = [text]
        new_vecs = self._encode(chunks)
        self.knowledge_base.extend(chunks)
        self.metadata_store.append(metadata if metadata else {})

        if self.embeddings is None:
            self.embeddings = new_vecs
        else:
            self.embeddings = torch.cat([self.embeddings, new_vecs], dim=0)

    def add_causal_relationship(
        self,
        cause: str,
        effect: str,
        strength: float = 1.0,
        condition: Optional[str] = None
    ):
        if condition:
            text_rep = f"If {condition}, because {cause}, then {effect}. (Strength: {strength:.2f})"
        else:
            text_rep = f"Because {cause}, then {effect}. (Strength: {strength:.2f})"

        self.add_knowledge(text_rep, metadata={
            "type": "causal",
            "cause": cause,
            "effect": effect,
            "strength": strength,
            "condition": condition
        })

    def add_triple(self, subj: str, pred: str, obj: str, metadata: Optional[Dict[str, Any]] = None):
        text_rep = f"{subj} {pred} {obj}"
        meta = metadata if metadata else {}
        meta.update({"type": "triple", "subject": subj,
                    "predicate": pred, "object": obj})
        self.add_knowledge(text_rep, metadata=meta)

    def search(self, query: str, k: int = 3) -> List[str]:
        if self.embeddings is None or len(self.knowledge_base) == 0:
            return []
        
        # クエリが空の場合は早期リターン
        if not query.strip():
            return []

        query_vec = self._encode([query])
        
        # 埋め込み次元の不一致チェックと修正（ランダム初期化時などの対策）
        if query_vec.shape[1] != self.embeddings.shape[1]:
            logger.warning(f"Dimension mismatch in search: query={query_vec.shape[1]}, db={self.embeddings.shape[1]}. Resetting embeddings.")
            # 簡易的な復旧策: 既存の埋め込みを再計算（時間がかかるが安全）
            if self.has_encoder: 
                self.embeddings = self._encode(self.knowledge_base)
            else:
                return [] # 復旧不能

        db_norm = F.normalize(self.embeddings, p=2, dim=1)
        q_norm = F.normalize(query_vec, p=2, dim=1)
        scores = torch.mm(q_norm, db_norm.transpose(0, 1)).squeeze(0)
        top_k = min(k, len(self.knowledge_base))
        if top_k == 0:
            return []
        indices = torch.topk(scores, k=top_k).indices
        return [self.knowledge_base[int(i)] for i in indices]

    # --- Persistence Methods (New) ---

    def save(self, directory: Optional[str] = None):
        """知識ベースと埋め込みを保存する"""
        target_dir = directory if directory else self.vector_store_path
        if not target_dir:
            logger.warning("⚠️ No directory specified for saving RAG data.")
            return

        os.makedirs(target_dir, exist_ok=True)
        
        # 1. テキストデータとメタデータの保存
        data = {
            "knowledge_base": self.knowledge_base,
            "metadata_store": self.metadata_store
        }
        with open(os.path.join(target_dir, "knowledge.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        # 2. 埋め込みテンソルの保存
        if self.embeddings is not None:
            torch.save(self.embeddings, os.path.join(target_dir, "embeddings.pt"))
        
        logger.info(f"💾 RAG knowledge saved to {target_dir} (Entries: {len(self.knowledge_base)})")

    def load(self, directory: str):
        """知識ベースと埋め込みを復元する"""
        if not os.path.exists(directory):
            return

        try:
            # 1. テキストデータの読み込み
            json_path = os.path.join(directory, "knowledge.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.knowledge_base = data.get("knowledge_base", [])
                    self.metadata_store = data.get("metadata_store", [])
            
            # 2. 埋め込みテンソルの読み込み
            emb_path = os.path.join(directory, "embeddings.pt")
            if os.path.exists(emb_path):
                self.embeddings = torch.load(emb_path, map_location="cpu")
            
            logger.info(f"📂 RAG knowledge loaded from {directory} (Entries: {len(self.knowledge_base)})")
        except Exception as e:
            logger.error(f"❌ Failed to load RAG data: {e}")