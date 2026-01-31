# ファイルパス: scripts/demos/system/run_chat_agent.py
# 日本語タイトル: Integrated Cognitive Chat Agent Demo
# 目的・内容:
#   - ユーザーと自然言語で対話するエージェントの実装。
#   - ArtificialBrainの全機能（意識、記憶、情動）を統合的に稼働させる。
#   - 会話を通じて「ユーザーの好み」などを記憶し、後のターンで想起・活用する様子をデモする。

import sys
import time
import logging
import torch
import random
from typing import Dict, Any, List
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

# ロギング設定 (コンソール出力を見やすくするため、INFOレベルのログはファイルに逃がす等の調整も可能だが、
# ここではユーザーとの対話をメインにするため、システムログは控えめにする)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChatAgent")

class CognitiveChatAgent:
    def __init__(self, brain: ArtificialBrain, user_name: str = "User"):
        self.brain = brain
        self.user_name = user_name
        self.brain.reset_state()
        
        # [Fix] 型注釈を追加
        self.conversation_history: List[str] = []
        
        # エージェントの性格設定 (情動パラメータに基づく修飾子)
        self.personas = {
            "neutral": "🤖 (Normal)",
            "curious": "👀 (Curious)",
            "bored": "😑 (Bored)",
            "happy": "😄 (Happy)",
            "afraid": "😨 (Scared)"
        }

    def perceive(self, user_input: str):
        """
        ユーザー入力を知覚し、脳のサイクルを回す。
        """
        print(f"\n👤 {self.user_name}: {user_input}")
        
        # 1. 記憶検索 (RAG): 入力に関連する過去の記憶があるか？
        retrieved_memories = self.brain.retrieve_knowledge(user_input)
        context_str = ""
        if retrieved_memories:
            context_str = f" [Memory: {retrieved_memories[0]}]"
            # 記憶がヒットしたら「驚き(Surprise)」を下げる（知っていることなので）
            surprise = 0.0
        else:
            # 知らないことなら「驚き」が高い
            surprise = 0.8

        # 2. 脳内処理 (Process Step)
        # 本来はテキストをEmbedding化して入力するが、ここではデモ用に
        # 入力文字列を直接扱い、内部状態の更新をメインに行う。
        brain_output = self.brain.process_step(user_input)
        
        # 3. 感情システムの更新 (手動補正)
        # process_step内でも更新されるが、会話のコンテキストに合わせて調整
        self.brain.motivation_system.process(user_input, prediction_error=surprise)
        
        return brain_output, retrieved_memories

    def generate_response(self, brain_output: Dict[str, Any], memories: list) -> str:
        """
        脳の内部状態（感情、活性度）に基づいて応答を生成する。
        ※ LLMではないため、テンプレートとルールベースで言語生成をシミュレートする。
        """
        drives = brain_output.get("drives", {})
        curiosity = drives.get("curiosity", 0.5)
        boredom = drives.get("boredom", 0.0)
        competence = drives.get("competence", 0.5)
        
        # 状態判定
        state = "neutral"
        if boredom > 0.7:
            state = "bored"
        elif curiosity > 0.7:
            state = "curious"
        elif competence > 0.8:
            state = "happy"
        
        prefix = self.personas[state]
        response = ""

        # A. 記憶に基づいた応答 (RAG Hit)
        if memories:
            memory_content = memories[0]
            if state == "curious":
                response = f"I remember you mentioned '{memory_content}'. Tell me more about it!"
            elif state == "bored":
                response = f"Yeah, yeah, '{memory_content}'. I know that already."
            else:
                response = f"I recall that: {memory_content}. Is that relevant now?"
        
        # B. 記憶がない場合の応答 (New Input)
        else:
            if state == "curious":
                response = "That's new to me! I've stored it in my memory. What else?"
            elif state == "bored":
                response = "I'm getting a bit sleepy... Tell me something exciting."
            elif state == "happy":
                response = "I'm feeling great! I've learned that."
            else:
                response = "I see. I've noted that down."

        # C. 思考内容の付加 (Workspaceの内容)
        thought = str(brain_output.get("conscious_broadcast", {}).get("source", "None"))
        debug_info = f" (Focus: {thought}, E: {self.brain.astrocyte.current_energy:.0f})"
        
        return f"{prefix} {response}{debug_info}"

    def memorize(self, user_input: str):
        """
        会話内容を長期記憶(RAG)に保存する。
        """
        # [Fix] RAGSystem.add_knowledge の第2引数は metadata(dict) なので修正
        self.brain.rag_system.add_knowledge(user_input, {"source": "user_chat", "raw": user_input})
        
    def sleep(self):
        print("\n💤 Agent is entering sleep mode to consolidate memories...")
        self.brain.sleep_cycle()
        print("🌅 Agent woke up refreshed!")

def run_chat_demo():
    print("\n" + "="*60)
    print("🧠 DORA System: Integrated Cognitive Chat Agent")
    print("="*60)
    print(" - Interaction: Natural Language")
    print(" - Brain Modules: SFormer, Global Workspace, Hippocampus, RAG, Motivation")
    print(" - Type 'sleep' to force memory consolidation.")
    print(" - Type 'quit' to exit.")
    print("-" * 60)

    # コンテナ初期化
    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    container.config.from_yaml(str(config_path))
    container.config.device.from_value("cpu")
    
    brain = container.artificial_brain()
    
    agent = CognitiveChatAgent(brain, user_name="LittleBuddha")
    
    # デモ用の初期知識
    print("📥 Initializing Knowledge Base...")
    brain.rag_system.add_knowledge("DORA is a neuromorphic AI project.", "DORA")
    brain.rag_system.add_knowledge("The sky is blue because of Rayleigh scattering.", "sky")
    
    print("✅ Ready to chat!\n")

    while True:
        try:
            user_input = input(">> ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit"]:
                print("👋 Shutting down DORA system.")
                break
            
            if user_input.lower() == "sleep":
                agent.sleep()
                continue

            # 1. 知覚 & 思考
            brain_output, memories = agent.perceive(user_input)
            
            # 2. 応答生成
            response = agent.generate_response(brain_output, memories)
            
            # 遅延演出 (思考時間)
            time.sleep(0.5)
            print(f"🤖 DORA: {response}")
            
            # 3. 学習 (記憶への書き込み)
            # 毎回記憶すると重複するので、記憶にヒットしなかった場合のみ保存するなどのロジックを入れる
            if not memories:
                agent.memorize(user_input)
                # 報酬を与える (新しいことを学ぶのは楽しい)
                brain.motivation_system.update_state({"reward": 0.5})
            else:
                # 既知の情報を繰り返されたら退屈する
                brain.motivation_system.update_state({"boredom": 0.1})

        except KeyboardInterrupt:
            print("\n👋 Force Quit.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("⚠️ Brain Error occurred. Resetting state...")
            brain.reset_state()

if __name__ == "__main__":
    run_chat_demo()