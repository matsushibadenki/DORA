# directory: scripts/experiments/learning
# file: run_plasticity_stability_test.py
# purpose: 即時学習が既存の知識に与える影響（安定性）を検証する

import torch
from torchvision import datasets
import sys
import os
import random
import numpy as np
from tqdm import tqdm

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

try:
    from snn_research.models.adapters.sara_adapter import SARAAdapter
except ImportError:
    sys.path.append(".")
    from snn_research.models.adapters.sara_adapter import SARAAdapter

def evaluate_accuracy(agent, dataset, limit=1000):
    """モデルの汎用精度を測定 (高速化のため1000件で近似)"""
    correct = 0
    total = 0
    indices = np.random.choice(len(dataset), min(limit, len(dataset)), replace=False)
    
    # プログレスバーなしで高速実行
    for idx in indices:
        img, label = dataset[idx]
        result = agent.think(img)
        if result["prediction"] == label:
            correct += 1
        total += 1
        
    return 100.0 * correct / total

def run_test():
    print("=== SARA Stability-Plasticity Test ===")
    
    model_path = "models/checkpoints/sara_mnist_v5.pth"
    if not os.path.exists(model_path):
        print("Please train the model first.")
        return

    agent = SARAAdapter(model_path, device="cpu")
    dataset = datasets.MNIST('../data', train=False, download=True)
    
    # 1. ベースライン測定
    print("\n[Phase 1] Measuring Baseline Health...")
    acc_before = evaluate_accuracy(agent, dataset)
    print(f"Initial Accuracy: {acc_before:.2f}%")
    
    # 2. ミスの特定と修正
    print("\n[Phase 2] Finding a weakness...")
    mistake_found = False
    target_img = None
    target_label = None
    
    # ランダムに探す
    max_search = 500
    for _ in range(max_search):
        idx = random.randint(0, len(dataset)-1)
        img, label = dataset[idx]
        res = agent.think(img)
        if res["prediction"] != label:
            mistake_found = True
            target_img = img
            target_label = label
            print(f"Mistake Found: Input {label} -> Pred {res['prediction']} (Conf: {res['confidence']*100:.1f}%)")
            break
    
    if not mistake_found:
        print("No high-confidence mistakes found quickly. Agent is too smart!")
        return

    # 即時学習 (Plasticity)
    print("Injecting new knowledge (Online Learning)...")
    loss = agent.learn_instance(target_img, target_label, max_steps=50)
    
    # その事例を覚えたか確認
    res_after = agent.think(target_img)
    print(f"Correction Result: Input {target_label} -> Pred {res_after['prediction']} (Conf: {res_after['confidence']*100:.1f}%)")
    
    if res_after['prediction'] != target_label:
        print("❌ Failed to learn the specific instance. Aborting stability test.")
        return

    # 3. 安定性確認 (Stability)
    print("\n[Phase 3] Checking for Brain Damage (Catastrophic Forgetting)...")
    acc_after = evaluate_accuracy(agent, dataset)
    print(f"Post-Learning Accuracy: {acc_after:.2f}%")
    
    diff = acc_after - acc_before
    print("-" * 40)
    print(f"Accuracy Change: {diff:+.2f}%")
    
    if diff < -5.0:
        print("⚠️ CRITICAL: Catastrophic Forgetting Detected! The agent sacrificed general knowledge.")
    elif diff < -1.0:
        print("⚠️ WARNING: Slight degradation in general knowledge.")
    else:
        print("✅ SUCCESS: Plasticity achieved with high stability.")

if __name__ == "__main__":
    run_test()