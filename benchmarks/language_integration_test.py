# benchmarks/language_integration_test.py
# Title: DORA Language Integration Test (Dict Inspector)
# Description: 
#   Shock Testで返される辞書のキーを表示し、脳の出力構造を明らかにする。

import sys
import os
import torch
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.language_cortex import LanguageCortex

def run_language_test():
    print("\n" + "="*60)
    print("🗣️ DORA Language Integration Test (Dict Inspector)")
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
    
    # 2. Connection Check (Shock Test)
    print("\n🔌 [System] Running Connection Check (Shock Test)...")
    try:
        dummy_input = torch.ones(1, 128) * 100.0
        output = brain.process_step(dummy_input)
        
        if isinstance(output, torch.Tensor):
            print(f"   -> Shock Response: {output.sum().item()} spikes (Tensor)")
        elif isinstance(output, dict):
            print(f"   -> Shock Response: Dict with keys {list(output.keys())}")
            # Try to sum everything to confirm activity
            total_activity = 0
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    total_activity += v.sum().item()
            print(f"   -> Total Activity in Dict: {total_activity}")
        else:
            print(f"   -> Shock Response: Unknown type {type(output)}")
            
    except Exception as e:
        print(f"   -> Shock Test FAILED: {e}")

    # 3. Initialize Language Cortex
    lang_cortex = LanguageCortex(brain)
    
    # 4. Test Cases
    test_sentences = [
        "Hello DORA.", 
        "FIRE!"
    ]
    
    for text in test_sentences:
        lang_cortex.process_text(text)

    # 5. Cleanup
    os_kernel.shutdown()
    print("\n" + "="*60)
    print("✅ Test Complete")
    print("="*60)

if __name__ == "__main__":
    run_language_test()