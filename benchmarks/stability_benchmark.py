# ファイルパス: snn_research/benchmarks/stability_benchmark.py
from snn_research.models.visual_cortex import VisualCortex
from snn_research.utils.observer import NeuromorphicObserver # Fixed import
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import random
from sklearn.linear_model import LogisticRegression  # type: ignore
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def flatten_tensor(x):
    return torch.flatten(x)

def run_benchmark(config):
    observer = NeuromorphicObserver(experiment_name="stability_benchmark")
    # ... (rest of the logic, change observer calls if necessary, but API is compatible)
    # Observer API: set_config, log, log_metric, save_results are compatible.
    observer.set_config(vars(config))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    observer.log(f"🚀 Starting Stability Benchmark on {device}")

    # ... (dataset loading ...)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    try:
        full_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    except:
        full_dataset = datasets.FakeData(transform=transform) # Fallback

    subset_size = 200
    train_dataset = Subset(full_dataset, range(subset_size))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)

    stability_scores = []

    for run in range(config.num_runs):
        seed = 42 + run
        set_seed(seed)
        model = VisualCortex(device=torch.device(device))
        model.to(device)

        for epoch in range(config.epochs):
            total_goodness_pos = 0.0
            total_goodness_neg = 0.0
            
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)
                _ = model(data, phase="wake")
                # Fix: Cast to float explicitly
                g_pos = float(model.get_goodness().get("V1_goodness", 0.0))
                total_goodness_pos += g_pos
                
                noise = torch.randn_like(data).to(device)
                _ = model(noise, phase="sleep")
                g_neg = float(model.get_goodness().get("V1_goodness", 0.0))
                total_goodness_neg += g_neg

            avg_pos = total_goodness_pos / len(train_loader)
            avg_neg = total_goodness_neg / len(train_loader)
            step = run * config.epochs + epoch
            observer.log_metric("goodness_pos", avg_pos, step)
            observer.log_metric("goodness_neg", avg_neg, step)
            observer.log(f"Run {run+1} Epoch {epoch+1}: Pos={avg_pos:.4f}, Neg={avg_neg:.4f}")

    observer.save_results()
    print("Benchmark Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    run_benchmark(args)