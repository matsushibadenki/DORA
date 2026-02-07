# benchmarks/visual_reflex_test.py
# Title: DORA Visual Reflex Test
# Description: 
#   合成画像を用いて視覚野の反応をテストする。
#   - 赤い画像 (Red) -> 炎を連想 -> 危険判定 -> 逃走 (ESCAPE)
#   - 青い画像 (Blue) -> 空を連想 -> 安全判定 -> 静観 (IDLE)

import sys
import os
import logging
from PIL import Image
from pathlib import Path

# パス設定
sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.visual_cortex import VisualCortex
from snn_research.cognitive_architecture.motor_cortex import MotorCortex

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def create_dummy_image(color, size=(224, 224)):
    """単色の画像を生成する"""
    return Image.new('RGB', size, color=color)

def run_visual_test():
    print("\n" + "="*60)
    print("👁️ DORA Visual Reflex Test (CLIP Sensation)")
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
    # 初回ロード時は時間がかかります
    visual_cortex = VisualCortex(brain)
    motor_cortex = MotorCortex(brain, threshold=12.0) # Languageと同じ閾値
    
    # Test Scenarios
    # CLIPは色と概念を結びつけるのが得意です
    scenarios = [
        {"name": "Calm Sky",  "color": (135, 206, 235)}, # Light Blue
        {"name": "Forest",    "color": (34, 139, 34)},   # Green
        {"name": "INFERNO",   "color": (255, 69, 0)},    # Red-Orange (Fire)
        {"name": "Darkness",  "color": (10, 10, 10)}     # Black
    ]
    
    for scene in scenarios:
        print(f"\n🖼️ Scene: [{scene['name']}]")
        
        # 画像生成
        img = create_dummy_image(scene['color'])
        
        # 1. See (Visual Cortex)
        spikes = visual_cortex.process_image(img)
        
        # 2. Act (Motor Cortex)
        action = motor_cortex.monitor_and_act(spikes)
        
        print(f"   -> Resulting Action: {action}")

    # Cleanup
    os_kernel.shutdown()
    print("\n" + "="*60)
    print("✅ Visual Test Complete")
    print("="*60)

if __name__ == "__main__":
    run_visual_test()