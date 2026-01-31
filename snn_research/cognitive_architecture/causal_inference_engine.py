# /snn_research/cognitive_architecture/causal_inference_engine.py
# 日本語タイトル: 因果推論エンジン (MyPy Fix)
# 目的: GlobalWorkspace.subscribe のコールバックシグネチャ不整合を修正。

from typing import Dict, Any, Optional, List, Callable
import logging
import re
from .rag_snn import RAGSystem
from .global_workspace import GlobalWorkspace

logger = logging.getLogger(__name__)


class DemocritusPipeline:
    """
    DEMOCRITUSシステムのパイプライン実装。
    """

    def __init__(self, generator_callback: Callable[[str], str]):
        self.generator = generator_callback

    def run_pipeline(self, text: str) -> List[Dict[str, Any]]:
        topics = self._extract_topics(text)
        if not topics:
            return []

        extracted_triples = []
        for topic in topics:
            questions = self._generate_causal_questions(text, topic)

            for question in questions:
                answer = self._get_model_response(
                    f"Context: {text}\nQuestion: {question}\nAnswer concisely:")
                triples = self._extract_triples_from_answer(
                    topic, question, answer)
                extracted_triples.extend(triples)

        return self._deduplicate_triples(extracted_triples)

    def _extract_topics(self, text: str) -> List[str]:
        prompt = (
            f"Analyze the following text and list up to 3 main scientific or logical topics mentioned.\n"
            f"Text: {text[:500]}...\n"
            f"Output format: Topic1, Topic2, Topic3"
        )
        response = self._get_model_response(prompt)
        topics = [t.strip() for t in response.split(',')]
        return [t for t in topics if t]

    def _generate_causal_questions(self, text: str, topic: str) -> List[str]:
        prompt = (
            f"Based on the text about '{topic}', generate 2 questions that ask about cause-and-effect relationships.\n"
            f"Start questions with 'What causes' or 'What is the effect of'.\n"
            f"Text: {text[:500]}..."
        )
        response = self._get_model_response(prompt)
        questions = [q.strip() for q in response.split('\n') if '?' in q]
        return questions[:2]

    def _extract_triples_from_answer(self, topic: str, question: str, answer: str) -> List[Dict[str, Any]]:
        prompt = (
            f"Extract causal triples from the statement below. \n"
            f"Format: [Subject] -> [Relation] -> [Object] (Strength: 0.0-1.0)\n"
            f"Statement: Since {answer}, it implies a causal link related to {topic}."
        )
        response = self._get_model_response(prompt)
        return self._parse_triple_response(response)

    def _parse_triple_response(self, response: str) -> List[Dict[str, Any]]:
        triples = []
        pattern = r"\[(.*?)\] -> \[(.*?)\] -> \[(.*?)\] \(Strength: (0\.\d+|1\.0)\)"
        matches = re.findall(pattern, response)

        for subj, rel, obj, strength in matches:
            triples.append({
                "subject": subj.strip(),
                "predicate": rel.strip(),
                "object": obj.strip(),
                "strength": float(strength),
                "type": "causal_triple"
            })
        return triples

    def _get_model_response(self, prompt: str) -> str:
        return self.generator(prompt).strip()

    def _deduplicate_triples(self, triples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique = {}
        for t in triples:
            key = f"{t['subject']}_{t['predicate']}_{t['object']}"
            if key not in unique:
                unique[key] = t
            else:
                if t['strength'] > unique[key]['strength']:
                    unique[key] = t
        return list(unique.values())


class CausalInferenceEngine:
    def __init__(
        self,
        rag_system: RAGSystem,
        workspace: GlobalWorkspace,
        inference_threshold: float = 0.6,
        llm_backend: Optional[Callable[[str], str]] = None
    ):
        self.rag_system = rag_system
        self.workspace = workspace
        self.inference_threshold = inference_threshold
        self.workspace.subscribe(self.handle_conscious_broadcast)

        self.pipeline = DemocritusPipeline(
            llm_backend if llm_backend else self._dummy_generator)

        logger.info(
            "🔥 CausalInferenceEngine (w/ DEMOCRITUS pipeline) initialized.")

    def set_llm_backend(self, generator: Callable[[str], str]) -> None:
        self.pipeline.generator = generator

    def _dummy_generator(self, prompt: str) -> str:
        logger.warning(
            "LLM backend not set for CausalInferenceEngine. Returning empty string.")
        return ""

    def process_text_for_causality(self, text: str, source: str = "perception") -> int:
        logger.info(f"🧪 Running DEMOCRITUS pipeline on text from {source}...")
        triples = self.pipeline.run_pipeline(text)

        for triple in triples:
            self._crystallize_causal_triple(triple, source)

        return len(triples)

    def _crystallize_causal_triple(self, triple: Dict[str, Any], context_source: str) -> None:
        subj = triple['subject']
        pred = triple['predicate']
        obj = triple['object']
        strength = triple.get('strength', 1.0)

        logger.info(
            f"💎 Causal Triple Crystallized: [{subj}] --{pred}--> [{obj}] (s={strength})")

        self.rag_system.add_triple(
            subj=subj,
            pred=pred,
            obj=obj,
            metadata={
                "source": context_source,
                "strength": strength,
                "paradigm": "DEMOCRITUS_LCM"
            }
        )

        if "cause" in pred.lower() or "lead" in pred.lower() or "result" in pred.lower():
            self.rag_system.add_causal_relationship(
                cause=subj,
                effect=obj,
                strength=strength,
                condition=f"via {pred}"
            )

        if strength > self.inference_threshold:
            self.workspace.upload_to_workspace(
                source_name="causal_inference_engine",
                content={
                    "type": "new_causal_discovery",
                    "triple": triple
                },
                salience=strength
            )

    def _crystallize_causality(self, context: str, cause: str, effect: str, strength: float) -> None:
        logger.info(
            f"🔥 Simple Causal Discovery: {cause} -> {effect} (strength={strength:.2f})")

        self.rag_system.add_causal_relationship(
            cause=cause,
            effect=effect,
            strength=strength
        )

        self.workspace.upload_to_workspace(
            source_name="causal_inference_engine",
            content={
                "type": "causal_credit",
                "cause": cause,
                "effect": effect,
                "strength": strength
            },
            salience=0.8
        )

    def handle_conscious_broadcast(self, broadcast_data: Dict[str, Any]) -> None:
        """
        意識に上った情報（Conscious Broadcast）を監視し、
        テキスト情報が含まれていれば因果抽出パイプラインを起動する。
        MyPy Fix: 引数を辞書1つに変更。
        """
        source = str(broadcast_data.get("source", "unknown"))
        
        # テキストデータが含まれているかチェック
        text_content = None

        if isinstance(broadcast_data, str):
            text_content = broadcast_data
        elif isinstance(broadcast_data, dict):
            if "text" in broadcast_data:
                text_content = broadcast_data["text"]
            elif "observation" in broadcast_data:
                text_content = broadcast_data["observation"]

        # 十分な長さがあればパイプラインを実行
        if text_content and isinstance(text_content, str) and len(text_content) > 50:
            self.process_text_for_causality(text_content, source=source)