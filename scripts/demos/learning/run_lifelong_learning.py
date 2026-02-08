# directory: scripts/demos/learning
# file: run_lifelong_learning.py
# purpose: SARAエージェントが間違いを指摘され、即座に学習して修正するデモ

import torch
from torchvision import datasets
import sys
import os
import random
import time

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    sys.path.append(".")
    from snn_research.models.adapters.sara_adapter import SARAAdapter

def run_learning_demo():
    print("=== SARA Online Plasticity Demo ===")
    
    model_path = "models/checkpoints/sara_mnist_v5.pth"
    if not os.path.exists(model_path):
        print("Please train the model first.")
        return

    agent = SARAAdapter(model_path, device="cpu")
    dataset = datasets.MNIST('../data', train=False, download=True)
    
    print("\n[Phase 1] Searching for a mistake...")
    
    mistake_found = False
    target_img = None
    target_label = None
    
    # 間違いが見つかるまでループ
    while not mistake_found:
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        
        result = agent.think(img)
        pred = result["prediction"]
        conf = result["confidence"]
        
        if pred != label:
            print(f"FOUND MISTAKE! Input: {label}, Predicted: {pred} (Conf: {conf*100:.1f}%)")
            mistake_found = True
            target_img = img
            target_label = label
        else:
            # 正解ならスキップ（画面が埋まるので表示しない）
            pass
            
    print("\n[Phase 2] Teaching the agent...")
    print(f"Human: 'No, that is actually a {target_label}.'")
    
    # 即時学習実行
    start = time.time()
    loss = agent.learn_instance(target_img, target_label)
    duration = (time.time() - start) * 1000
    
    print(f"Agent: 'Understood. Updating synaptic weights...' (Loss: {loss:.4f}, Time: {duration:.1f}ms)")
    
    print("\n[Phase 3] Verification...")
    # 同じ画像を見せる
    result_new = agent.think(target_img)
    pred_new = result_new["prediction"]
    conf_new = result_new["confidence"]
    
    if pred_new == target_label:
        print(f"Agent: 'Now I see it! It's a {pred_new}.' (Conf: {conf_new*100:.1f}%)")
        print("✅ SUCCESS: Instant learning achieved.")
    else:
        print(f"Agent: 'I still think it's {pred_new}...'")
        print("❌ FAILURE: Needs more training.")

if __name__ == "__main__":
    run_learning_demo()