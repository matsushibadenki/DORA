# directory: scripts/run_production.py
# title: SARA Production Run (Fixed Import)
# description: import numpy as np を追加した修正版実行スクリプト。

import sys
import os
import time
import argparse
import numpy as np  # これを追加しました

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from snn_research.systems.sara_system import SaraSystem

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    print("Need torchvision.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--save_path", type=str, default="models/checkpoints/sara_model_v1.pkl")
    args = parser.parse_args()
    
    print(f"--- SARA Production Run ---")
    system = SaraSystem()
    
    # Data Loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST('./data', train=False, transform=transform)
    
    # Training Loop
    print(f"Training on {args.samples} samples...")
    start_time = time.time()
    
    indices = np.random.choice(len(train_data), args.samples, replace=False)
    
    for i, idx in enumerate(indices):
        img, target = train_data[idx]
        system.train_sample(img.numpy().flatten(), target)
        
        if (i+1) % 1000 == 0:
            print(f"Processed {i+1} samples...")
            
    print(f"Training Time: {time.time() - start_time:.1f}s")
    
    # Sleep Phase
    system.sleep()
    
    # Evaluation
    print("Evaluating on 500 test samples...")
    correct = 0
    test_indices = np.random.choice(len(test_data), 500, replace=False)
    
    for idx in test_indices:
        img, target = test_data[idx]
        pred = system.predict_sample(img.numpy().flatten())
        if pred == target:
            correct += 1
            
    acc = correct / 500 * 100
    print(f"Test Accuracy: {acc:.2f}%")
    
    # Save
    system.save(args.save_path)
    print(f"Model saved to {args.save_path}")

if __name__ == "__main__":
    main()