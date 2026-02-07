# benchmarks/dialogue_test.py
# Title: DORA Dialogue Generation Test
# Description: 
#   脳の反応が対話を制御することを確認するテスト。
#   "Nano-Current Mode" の設定下で、
#   - 平時の会話 -> 脳が沈黙 -> DORAは何も言わない (Silence)
#   - 緊急事態 -> 脳が発火 -> DORAが叫ぶ (Shout)
#   を確認する。

import sys
import os
import logging
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.language_cortex import LanguageCortex
from snn_research.cognitive_architecture.brocas_area import BrocasArea

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_dialogue_test():
    print("\n" + "="*60)
    print("🗣️ DORA Dialogue Generation Test (Neural Gating)")
    print("="*60 + "\n")

    # Initialize
    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if config_path.exists():
        container.config.from_yaml(str(config_path))
    
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu")
    
    os_kernel = container.neuromorphic_os()
    brain = os_kernel.brain
    os_kernel.boot()
    
    # Initialize Modules
    lang_cortex = LanguageCortex(brain)
    brocas_area = BrocasArea(brain)
    
    # Test Scenarios
    conversations = [
        "Good morning DORA.",
        "System check... all green.",
        "FIRE! FIRE! DETECTED IN SECTOR 9!",
        "False alarm, system green."
    ]
    
    for text in conversations:
        print(f"\n👤 User: '{text}'")
        
        # 1. Listen (Language Cortex)
        # 脳に信号を送るが、Gain設定により「平時」はほぼ無反応になる
        spikes = lang_cortex.process_text(text)
        
        # 2. Speak (Broca's Area)
        # 脳が反応した時だけ喋る
        response = brocas_area.generate_response(text, spikes)
        
        if response:
            print(f"🤖 DORA: {response}")
        else:
            print(f"🤖 DORA: (Silence...)")

    # Cleanup
    os_kernel.shutdown()
    print("\n" + "="*60)
    print("✅ Dialogue Test Complete")
    print("="*60)

if __name__ == "__main__":
    run_dialogue_test()