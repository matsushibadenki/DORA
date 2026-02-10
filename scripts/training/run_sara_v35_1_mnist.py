# directory: scripts/training/run_sara_v35_1_mnist.py
# title: SARA v35.1 Training - Liquid Harmony
# description: 再帰結合を持つ真のLSMを用いた大規模学習スクリプト（チューニング版）。

import sys
import os
import numpy as np
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from snn_research.models.experimental.sara_v35_1_engine import SaraEngineV35_1

try:
    import torch
    from torchvision import datasets, transforms
except ImportError:
    print("Need torchvision.")
    sys.exit(1)

def get_mnist_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test = datasets.MNIST('./data', train=False, transform=transform)
    return train, test

def img_to_poisson(img_flat, time_steps=50):
    img_flat = np.maximum(0, img_flat)
    img_flat = img_flat / (np.max(img_flat) + 1e-6)
    rate = img_flat * 0.4
    spike_train = []
    for _ in range(time_steps):
        fired = np.where(np.random.rand(len(img_flat)) < rate)[0].tolist()
        spike_train.append(fired)
    return spike_train

def evaluate(engine, dataset, n_samples=300, steps=50):
    correct = 0
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    for idx in indices:
        img, target = dataset[idx]
        spike_train = img_to_poisson(img.numpy().flatten(), steps)
        pred = engine.predict(spike_train)
        if pred == target:
            correct += 1
    return correct / n_samples * 100

def main():
    parser = argparse.ArgumentParser()
    # サンプル数を20000に増量！
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--samples", type=int, default=20000)
    args = parser.parse_args()

    input_size = 784
    output_size = 10
    time_steps = 50
    
    print(f"Initializing SARA v35.1 (Liquid Harmony, 20000 samples)...")
    engine = SaraEngineV35_1(input_size, output_size)
    
    train_data, test_data = get_mnist_data()
    start_total = time.time()
    
    for epoch in range(args.epochs):
        indices = np.random.choice(len(train_data), args.samples, replace=False)
        
        for i, idx in enumerate(indices):
            img, target = train_data[idx]
            spike_train = img_to_poisson(img.numpy().flatten(), time_steps)
            
            engine.train_step(spike_train, target, dropout_rate=0.1)
            
            if i == 0:
                stats = engine.get_activity_stats(spike_train)
                print(f"Reservoir Activity: Fast={stats[0]:.1f}, Med={stats[1]:.1f}, Slow={stats[2]:.1f}")
            
            if (i+1) % 1000 == 0:
                print(f"Epoch {epoch+1}: {i+1}/{args.samples} processed.")
        
        engine.sleep_phase(prune_rate=0.05)
        
        val_acc = evaluate(engine, test_data, n_samples=500, steps=time_steps)
        print(f"Epoch {epoch+1} Val Acc: {val_acc:.2f}%")

    print(f"Total Time: {time.time() - start_total:.1f}s")
    final_acc = evaluate(engine, test_data, n_samples=1000, steps=time_steps)
    print(f"Final Test Acc: {final_acc:.2f}%")

if __name__ == "__main__":
    main()