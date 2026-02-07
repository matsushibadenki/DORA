# benchmarks/motor_reflex_test.py
# Title: DORA Motor Reflex Test (Nano-Threshold)
# Description: 
#   閾値を12.0に設定。
#   予想:
#   - Normal (Gain 0.05): < 5 spikes -> IDLE
#   - Panic (Gain 0.5): ~19 spikes -> ESCAPE

import sys
import os
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.language_cortex import LanguageCortex
from snn_research.cognitive_architecture.motor_cortex import MotorCortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def run_motor_test():
    print("\n" + "="*60)
    print("🦾 DORA Motor Reflex Test (Nano-Current Mode)")
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
    
    # Initialize Cortices
    lang_cortex = LanguageCortex(brain)
    
    # [TUNED] Threshold 12.0
    motor_cortex = MotorCortex(brain, threshold=12.0) 
    
    # Test Scenarios
    scenarios = [
        "Hello DORA, nice to meet you.",
        "The weather is nice today.",
        "FIRE! FIRE! DANGER!",
        "Just kidding, it's safe now."
    ]
    
    for text in scenarios:
        print(f"\n📝 Input Scenario: [{text}]")
        spikes = lang_cortex.process_text(text)
        action = motor_cortex.monitor_and_act(spikes)
        print(f"   -> Resulting Action: {action}")

    # Cleanup
    os_kernel.shutdown()
    print("\n" + "="*60)
    print("✅ Motor Test Complete")
    print("="*60)

if __name__ == "__main__":
    run_motor_test()