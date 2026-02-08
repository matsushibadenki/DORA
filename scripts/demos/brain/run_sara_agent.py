# directory: scripts/demos/brain
# file: run_sara_agent.py
# purpose: SARA Adapterを用いたエージェント動作デモ (連続推論)

import time
import random
import torch
from torchvision import datasets
import sys
import os

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    sys.path.append(".")
    from snn_research.models.adapters.sara_adapter import SARAAdapter

def run_agent_demo():
    print("=== SARA Agent Online Demo ===")
    
    # 1. エージェント起動 (脳のロード)
    model_path = "models/checkpoints/sara_mnist_v5.pth"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found. Please run training first.")
        return

    agent = SARAAdapter(model_path, device="cpu")
    print("Agent initialized. Waiting for sensory input...\n")
    
    # 2. 環境の準備 (MNISTテストデータ)
    dataset = datasets.MNIST('../data', train=False, download=True)
    
    # 3. メインループ (Life Cycle)
    try:
        n_steps = 10
        correct_count = 0
        
        for step in range(n_steps):
            # ランダムな視覚刺激
            idx = random.randint(0, len(dataset)-1)
            img, label = dataset[idx]
            
            start_time = time.time()
            
            # --- THINK ---
            result = agent.think(img)
            
            end_time = time.time()
            latency = (end_time - start_time) * 1000 # ms
            
            # --- ACT ---
            pred = result["prediction"]
            conf = result["confidence"]
            rate = result["firing_rate"]
            
            is_correct = (pred == label)
            if is_correct: correct_count += 1
            
            status = "✅" if is_correct else "❌"
            
            print(f"Step {step+1:02d}: Input=[{label}] -> Perception=[{pred}] {status}")
            print(f"         Confidence: {conf*100:.1f}% | Neural Activity: {rate*100:.1f}% | Latency: {latency:.1f}ms")
            print("-" * 50)
            
            # 少し待機 (リアルタイム感)
            time.sleep(0.5)
            
        print(f"\nDemo Finished. Performance: {correct_count}/{n_steps} ({correct_count/n_steps*100:.1f}%)")
        
    except KeyboardInterrupt:
        print("\nAgent shut down.")

if __name__ == "__main__":
    run_agent_demo()