# directory: scripts/run_inference.py
# title: SARA Inference Demo
# description: 保存されたモデル(models/checkpoints/sara_model_v1.pkl)をロードし、未知のデータに対して推論を行うデモ。

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# パス設定
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from snn_research.systems.sara_system import SaraSystem

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    print("Need torchvision.")
    sys.exit(1)

def main():
    model_path = "models/checkpoints/sara_model_v1.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    # 学習済みモデルをロード（初期化学習は不要）
    system = SaraSystem(model_path=model_path)
    print("Model loaded successfully.")

    # テストデータの読み込み
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    # ランダムに5枚選んでテスト
    indices = np.random.choice(len(test_data), 5, replace=False)
    
    print("\n--- Inference Results ---")
    for idx in indices:
        img, target = test_data[idx]
        img_flat = img.numpy().flatten()
        
        # 推論実行
        prediction = system.predict_sample(img_flat)
        
        result = "✅ Correct" if prediction == target else "❌ Wrong"
        print(f"Image ID: {idx} | True Label: {target} | SARA Prediction: {prediction} -> {result}")

if __name__ == "__main__":
    main()