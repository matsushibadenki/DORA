# ファイルパス: snn_research/benchmarks/stability_benchmark.py
# 日本語タイトル: Visual Cortex Stability Benchmark (Enhanced)
# 目的: Measure learning stability and reproducibility with detailed metrics

from snn_research.models.visual_cortex import VisualCortex
from snn_research.utils.observer import ExperimentObserver
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
import argparse
import random
from sklearn.linear_model import LogisticRegression  # type: ignore
import sys

# Add project root to path
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../')))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def flatten_tensor(x):
    return torch.flatten(x)


def run_benchmark(config):
    observer = ExperimentObserver(experiment_name="stability_benchmark")
    observer.set_config(vars(config))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    observer.log(f"🚀 Starting Stability Benchmark on {device}")

    # Data Preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '../../data')
    try:
        full_dataset = datasets.MNIST(
            data_dir, train=True, download=True, transform=transform)
    except Exception as e:
        observer.log(f"Failed to load dataset: {e}", "error")
        return

    # Use a subset for faster stability check
    subset_size = 2000
    train_dataset = Subset(full_dataset, range(subset_size))
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True)

    stability_scores = []

    for run in range(config.num_runs):
        seed = 42 + run
        set_seed(seed)
        observer.log(f"--- Run {run+1}/{config.num_runs} (Seed: {seed}) ---")

        model = VisualCortex(device=device)
        model.to(device)

        # Training Loop
        for epoch in range(config.epochs):
            total_goodness_pos = 0.0
            total_goodness_neg = 0.0

            # Epoch metrics storage
            epoch_goodness_pos = []
            epoch_goodness_neg = []

            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(device)

                # --- Wake Phase (Positive) ---
                # Data driven
                _ = model(data, phase="wake")
                # Collect stats
                g_pos = model.get_goodness().get("V1_goodness", 0)
                total_goodness_pos += g_pos
                epoch_goodness_pos.append(g_pos)

                # --- Sleep Phase (Negative) ---
                # Noise driven (Dreaming)
                noise = torch.randn_like(data).to(device)
                _ = model(noise, phase="sleep")
                g_neg = model.get_goodness().get("V1_goodness", 0)
                total_goodness_neg += g_neg
                epoch_goodness_neg.append(g_neg)

            # Log Epoch Metrics
            avg_pos = total_goodness_pos / len(train_loader)
            avg_neg = total_goodness_neg / len(train_loader)

            step = run * config.epochs + epoch
            observer.log_metric("goodness_pos", avg_pos, step)
            observer.log_metric("goodness_neg", avg_neg, step)

            observer.log(
                f"Run {run+1} Epoch {epoch+1}: Pos={avg_pos:.4f}, Neg={avg_neg:.4f}")

            # Stability Check
            if avg_pos > 1000 or avg_pos < 0.001:
                observer.log(
                    "⚠️ Instability Detected (Energy Explosion/Vanishing)", "warning")

        # --- Post-Run Evaluation: Linear Probe ---
        # Extract features and train a simple classifier to check feature quality
        observer.log("Running Linear Probe Evaluation...")
        X_train_features = []
        y_train_labels = []

        # Use a small subset for probe training to save time
        probe_loader = DataLoader(Subset(full_dataset, range(
            2000, 3000)), batch_size=64, shuffle=False)

        model.eval()
        with torch.no_grad():
            for data, target in probe_loader:
                data = data.to(device)
                # Forward pass without learning
                # phase="test" avoids learning updates
                _ = model(data, phase="test")
                # Get internal representation (e.g., V1 spikes or mem)
                # VisualCortex doesn't expose generic feature getter easily, using substrate prev_spikes
                spikes = model.substrate.prev_spikes.get("V1")
                if spikes is not None:
                    X_train_features.append(spikes.cpu().numpy())
                    y_train_labels.append(target.numpy())

        if X_train_features:
            X_train = np.concatenate(X_train_features)
            y_train = np.concatenate(y_train_labels)

            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(X_train, y_train)

            train_acc = clf.score(X_train, y_train)
            observer.log(f"Run {run+1} Linear Probe Accuracy: {train_acc:.4f}")
            observer.log_metric("probe_accuracy", train_acc, run)
            stability_scores.append(train_acc)
        else:
            observer.log(
                "Failed to extract features for linear probe", "error")

    # Final Wrap-up
    observer.save_results()
    observer.plot_learning_curve(
        metric_names=["goodness_pos", "goodness_neg"], title="Goodness Stability")
    observer.plot_learning_curve(
        metric_names=["probe_accuracy"], title="Feature Quality (Linear Probe)")

    avg_score = sum(stability_scores) / \
        len(stability_scores) if stability_scores else 0.0
    observer.log(
        f"✅ Benchmark Completed. Avg Stability Score: {avg_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    run_benchmark(args)
