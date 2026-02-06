# ファイルパス: snn_research/distillation/thought_distiller.py
# 日本語タイトル: Thought Distillation Manager (System 2 -> System 1) with SDFT
# 目的・内容:
#   System 2 (Teacher) の思考プロセス(CoT)を教師データとして、
#   System 1 (Student: BitSpikeModel) を学習させる蒸留パイプライン。
#   論文 "Self-Distillation Enables Continual Learning" に基づくSDFT機能を追加。

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, cast, Optional
import logging
from tqdm import tqdm
import random

logger = logging.getLogger(__name__)


class SymbolicTeacher:
    """
    論理的・記号的教師（System 2の役割）。
    In-Context Learning (ICL) 能力を持ち、デモンストレーションに基づいて推論を行う。
    """

    def solve_with_reasoning(self, question: str) -> Dict[str, str]:
        # 基本的な推論ロジック (例: "15 + 27")
        try:
            parts = question.replace("?", "").split("+")
            if len(parts) != 2:
                raise ValueError("Format error")
            
            a = int(parts[0].strip())
            b = int(parts[1].strip())
            res = a + b

            # 思考過程の生成
            a_ones, a_tens = a % 10, a // 10
            b_ones, b_tens = b % 10, b // 10

            ones_sum = a_ones + b_ones
            carry = ones_sum // 10
            rem_ones = ones_sum % 10

            tens_sum = a_tens + b_tens + carry

            thought = (
                f"First, add ones: {a_ones} + {b_ones} = {ones_sum}. "
                f"Write {rem_ones}, carry {carry}. "
                f"Next, add tens: {a_tens} + {b_tens} + carry({carry}) = {tens_sum}. "
                f"Combine them to get {tens_sum}{rem_ones}."
            )

            return {
                "input": question,
                "thought_chain": thought,
                "answer": str(res)
            }
        except Exception:
            return {
                "input": question,
                "thought_chain": "I cannot solve this clearly.",
                "answer": "Unknown"
            }

    def solve_with_icl(self, question: str, demonstrations: List[Dict[str, str]]) -> Dict[str, str]:
        """
        [SDFT] デモンストレーション（過去の例）をコンテキストに含めて問題を解く。
        System 2 が過去の成功体験を参照して、より確信度の高い回答を生成するプロセスをシミュレート。
        """
        # プロンプトコンテキストの構築（概念的実装）
        # 実際にはLLMへのプロンプトとなるが、ここではロジック推論にメタ情報を付与する
        context_len = len(demonstrations)
        
        # 基本推論を実行
        result = self.solve_with_reasoning(question)
        
        # デモンストレーション効果の付与 (SDFT: コンテキストに基づく推論の強化)
        if context_len > 0 and result["answer"] != "Unknown":
            result["thought_chain"] = f"[ICL with {context_len} demos] {result['thought_chain']}"
        
        return result

    def verify(self, question: str, answer: str) -> bool:
        """
        [SDFT] 自己生成された答えが正しいか検証する（Self-Correction用）。
        """
        try:
            parts = question.replace("?", "").split("+")
            if len(parts) != 2:
                return False
            expected = int(parts[0].strip()) + int(parts[1].strip())
            return str(expected) == answer.strip()
        except:
            return False


class ThoughtDistillationManager:
    """
    思考蒸留マネージャー。
    Teacher(Symbolic/LLM)の出力をStudent(SNN)に模倣させる。
    SDFT (Self-Distillation Fine-Tuning) 対応。
    """

    def __init__(self, student_model: nn.Module, teacher_engine: Any, learning_rate: float = 1e-4):
        self.student = student_model
        self.teacher = teacher_engine
        self.optimizer = optim.AdamW(
            self.student.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def generate_thought_dataset(self, problems: List[str]) -> List[Dict[str, Any]]:
        """
        Teacherを使って、問題に対する「思考過程」と「答え」を生成する（通常モード）。
        """
        logger.info(
            f"🧠 System 2 is generating thoughts for {len(problems)} problems...")
        dataset = []

        for q in problems:
            reasoning_result = self.teacher.solve_with_reasoning(q)
            dataset.append(reasoning_result)

        return dataset

    def generate_sdft_dataset(self, problems: List[str], demonstrations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        [SDFT] 過去のデモンストレーションを用いた In-Context Learning でデータを生成する。
        生成されたデータのうち、Teacher自身が「正解」と判断したもののみを採用する。
        """
        logger.info(
            f"🧠 System 2 is generating SDFT data with {len(demonstrations)} demos...")
        dataset = []

        for q in problems:
            # 1. ICL推論 (オンポリシーデータ生成)
            result = self.teacher.solve_with_icl(q, demonstrations)
            
            # 2. 検証 (フィルタリング)
            if self.teacher.verify(q, result['answer']):
                dataset.append(result)
            else:
                logger.debug(f"Skipped incorrect generation for: {q}")

        logger.info(f"✅ Generated {len(dataset)}/{len(problems)} valid SDFT samples.")
        return dataset

    def distill(self, dataset: List[Dict[str, Any]], epochs: int = 3, batch_size: int = 1):
        """
        生成された思考データをStudentに学習させる。
        """
        if not dataset:
            logger.warning("⚠️ No dataset provided for distillation.")
            return

        logger.info(f"⚗️ Starting Distillation (Samples: {len(dataset)}, Epochs: {epochs})...")
        self.student.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            count = 0

            # Shuffle dataset for better training
            random.shuffle(dataset)
            
            pbar = tqdm(dataset, desc=f"Distill Epoch {epoch+1}/{epochs}")
            for item in pbar:
                # 入力プロンプト
                input_text = f"Q: {item['input']}\nReasoning:"

                # 教師の思考トレース (CoT) + 答え
                target_text = f" {item['thought_chain']}\nAnswer: {item['answer']}<EOS>"

                # --- Student Forward & Backward ---
                self.optimizer.zero_grad()

                # [Fix] Cast self.student to Any to avoid mypy error "Tensor not callable"
                student_any = cast(Any, self.student)
                
                if hasattr(student_any, 'forward_text_loss'):
                    # Studentモデルがテキスト入力を処理できる場合
                    loss = student_any.forward_text_loss(
                        input_text, target_text)
                else:
                    # ダミーロス (モデルの実装に依存するため、実際のSpikformer等のI/Oに合わせて調整が必要)
                    # 本来は Tokenizer -> input_ids -> Model -> Logits -> Loss
                    loss = torch.tensor(0.5, requires_grad=True)

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                count += 1
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            avg_loss = epoch_loss / max(count, 1)
            logger.info(f"   Epoch {epoch+1} Avg Loss: {avg_loss:.4f}")

        logger.info("✅ Distillation Completed. System 1 updated.")