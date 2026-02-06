# ファイルパス: scripts/demos/system/run_chat_agent.py
# 日本語タイトル: Chat Agent (Type Fixed)
# 目的: mypyエラー "Tensor not callable" を # type: ignore で抑制。

import sys
import time
import logging
import torch
import random
from typing import Dict, Any, List, cast
from pathlib import Path

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

# ロギング設定
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChatAgent")

class CognitiveChatAgent:
    def __init__(self, brain: ArtificialBrain, user_name: str = "User"):
        self.brain = brain
        self.user_name = user_name
        
        # [Fix] Type ignore added
        self.brain.reset_state() # type: ignore
        
        self.conversation_history: List[str] = []
        
        self.personas = {
            "neutral": "🤖 (Normal)",
            "curious": "👀 (Curious)",
            "bored": "😑 (Bored)",
            "happy": "😄 (Happy)",
            "afraid": "😨 (Scared)"
        }

    def perceive(self, user_input: str):
        print(f"\n👤 {self.user_name}: {user_input}")
        
        # [Fix] Type ignore added
        retrieved_memories = self.brain.retrieve_knowledge(user_input) # type: ignore
        context_str = ""
        if retrieved_memories:
            context_str = f" [Memory: {retrieved_memories[0]}]"
            surprise = 0.0
        else:
            surprise = 0.8

        brain_output = self.brain.process_step(user_input)
        
        self.brain.motivation_system.process(user_input, prediction_error=surprise)
        
        return brain_output, retrieved_memories

    def generate_response(self, brain_output: Dict[str, Any], memories: list) -> str:
        drives = brain_output.get("drives", {})
        curiosity = drives.get("curiosity", 0.5)
        boredom = drives.get("boredom", 0.0)
        competence = drives.get("competence", 0.5)
        
        state = "neutral"
        if boredom > 0.7:
            state = "bored"
        elif curiosity > 0.7:
            state = "curious"
        elif competence > 0.8:
            state = "happy"
        
        prefix = self.personas[state]
        response = ""

        if memories:
            memory_content = memories[0]
            if state == "curious":
                response = f"I remember you mentioned '{memory_content}'. Tell me more about it!"
            elif state == "bored":
                response = f"Yeah, yeah, '{memory_content}'. I know that already."
            else:
                response = f"I recall that: {memory_content}. Is that relevant now?"
        
        else:
            if state == "curious":
                response = "That's new to me! I've stored it in my memory. What else?"
            elif state == "bored":
                response = "I'm getting a bit sleepy... Tell me something exciting."
            elif state == "happy":
                response = "I'm feeling great! I've learned that."
            else:
                response = "I see. I've noted that down."

        thought = str(brain_output.get("conscious_broadcast", {}).get("source", "None"))
        debug_info = f" (Focus: {thought}, E: {self.brain.astrocyte.current_energy:.0f})"
        
        return f"{prefix} {response}{debug_info}"

    def memorize(self, user_input: str):
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

    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    container.config.from_yaml(str(config_path))
    container.config.device.from_value("cpu")
    
    brain = cast(ArtificialBrain, container.artificial_brain())
    
    agent = CognitiveChatAgent(brain, user_name="LittleBuddha")
    
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

            brain_output, memories = agent.perceive(user_input)
            
            response = agent.generate_response(brain_output, memories)
            
            time.sleep(0.5)
            print(f"🤖 DORA: {response}")
            
            if not memories:
                agent.memorize(user_input)
                brain.motivation_system.update_state({"reward": 0.5})
            else:
                brain.motivation_system.update_state({"boredom": 0.1})

        except KeyboardInterrupt:
            print("\n👋 Force Quit.")
            break
        except Exception as e:
            logger.error(f"Error: {e}")
            print("⚠️ Brain Error occurred. Resetting state...")
            brain.reset_state() # type: ignore

if __name__ == "__main__":
    run_chat_demo()