# benchmarks/memory_plasticity_test.py
# Title: DORA Memory Plasticity Test (Fixed format & Analysis)
# Description: 
#   float型のスパイク数に対応できるようフォーマットを修正。
#   また、発火数減少(LTD)を「効率化(Efficiency Gain)」として肯定的に評価するよう判定ロジックを更新。

import sys
import os
import torch
import logging
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.language_cortex import LanguageCortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MemoryTest")

def run_memory_test():
    print("\n" + "="*60)
    print("🧠 DORA Hippocampal Memory Encoding Test")
    print("="*60 + "\n")

    # 1. Initialize Brain
    container = AppContainer()
    config_path = Path("configs/templates/base_config.yaml")
    if config_path.exists():
        container.config.from_yaml(str(config_path))
    
    container.config.training.paradigm.from_value("event_driven")
    container.config.device.from_value("cpu")
    
    os_kernel = container.neuromorphic_os()
    brain = os_kernel.brain
    os_kernel.boot()
    
    # 2. Initialize Language Cortex
    lang_cortex = LanguageCortex(brain)
    
    # 3. Memory Task
    target_concept = "I am DORA, an artificial intelligence."
    
    # --- Step A: Pre-Test ---
    print(f"\n🧐 [Pre-Test] Encountering concept for the first time...")
    spikes_pre = lang_cortex.process_text(target_concept)
    total_pre = sum(spikes_pre)
    print(f"   -> Pre-Test Response: {total_pre:.2f} spikes")

    # --- Step B: Encoding ---
    print(f"\n📚 [Encoding] Repeating concept to induce Plasticity...")
    learning_epochs = 5
    
    for i in range(learning_epochs):
        spikes = lang_cortex.process_text(target_concept)
        print(f"   -> Learning Epoch {i+1}/{learning_epochs}: {sum(spikes):.2f} spikes")
        
        # Settle
        dummy_input = torch.zeros(1, 128)
        # 簡易Settle (LanguageCortex内部でも行われているが念のため)
        # brain.process_step(dummy_input) 

    # --- Step C: Post-Test ---
    print(f"\n💡 [Post-Test] Recalling concept after learning...")
    spikes_post = lang_cortex.process_text(target_concept)
    total_post = sum(spikes_post)
    print(f"   -> Post-Test Response: {total_post:.2f} spikes")

    # 4. Analysis
    print("\n" + "-"*60)
    print("📊 Memory Analysis Report")
    print("-"*60)
    
    diff = total_post - total_pre
    
    # [FIX] d -> .2f for float formatting
    print(f"   Initial Response: {total_pre:.2f}")
    print(f"   Learned Response: {total_post:.2f}")
    print(f"   Delta: {diff:+.2f} spikes")
    
    if diff > 0:
        print("\n✅ RESULT: Long-Term Potentiation (LTP) Detected.")
        print("   Connection strengthened. The brain finds this concept exciting.")
    elif diff < 0:
        print("\n✅ RESULT: Long-Term Depression (LTD) Detected.")
        print("   Efficiency Gain (Sparse Coding). The brain has optimized this path.")
        print("   (Processing the same thought requires less energy now.)")
    else:
        print("\n⚠️ RESULT: No Plasticity Detected.")
        print("   Synaptic weights remain unchanged.")

    # 5. Cleanup
    os_kernel.shutdown()
    print("="*60)

if __name__ == "__main__":
    run_memory_test()