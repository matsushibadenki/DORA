# directory: scripts/demos/life
# file: run_sara_lifecycle.py
# purpose: SARAエージェントの活動と睡眠のサイクルデモ

import torch
from torchvision import datasets
import sys
import os
import random
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    sys.path.append(".")
    from snn_research.models.adapters.sara_adapter import SARAAdapter

def run_lifecycle():
    print("=== SARA Life Cycle Demo (Day & Night) ===")
    
    agent = SARAAdapter("models/checkpoints/sara_mnist_v5.pth", device="cpu")
    dataset = datasets.MNIST('../data', train=False, download=True)
    
    # --- DAYTIME (Activity) ---
    print("\n☀️ DAYTIME: Exploring and Learning")
    
    mistakes_found = 0
    max_activities = 5
    
    # 意図的にいくつかのミスを探して教える
    search_limit = 100
    for i in range(search_limit):
        if mistakes_found >= max_activities: break
        
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        
        res = agent.think(img)
        if res["prediction"] != label:
            print(f"[{i}] Mistake! Input {label} -> Pred {res['prediction']}. Teaching...")
            agent.learn_instance(img, label)
            mistakes_found += 1
            
    print(f"\nDay ended. Agent gathered {mistakes_found} new experiences in episodic memory.")
    
    # --- NIGHTTIME (Sleep) ---
    print("\n🌙 NIGHTTIME: Sleeping...")
    time.sleep(1)
    agent.sleep(epochs=5)
    
    # --- NEXT MORNING (Validation) ---
    print("\n☀️ NEXT MORNING: Verification")
    # ここで昨日覚えたことが定着しているか確認できる（デモ上は省略）
    print("Agent feels refreshed and smarter.")

if __name__ == "__main__":
    run_lifecycle()