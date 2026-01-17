from snn_research.models.visual_cortex import VisualCortex
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
import json
import logging
from datetime import datetime

# Path setup
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StabilityBenchmark")


def flatten_tensor(x):
    return torch.flatten(x)


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10, device: str = "cpu"):
    x_mod = x.clone()
    y_oh = F.one_hot(y, num_classes).float().to(device)
    scale_factor = 2.5
    if x_mod.dim() == 2:
        x_mod[:, :num_classes] = y_oh * scale_factor
    return x_mod


def validate(model, loader, device, config):
    correct = 0
    total = 0

    # Use standard SNN time usage

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            goodness_matrix = torch.zeros(batch_size, 10, device=device)

            for label in range(10):
                y_candidate = torch.full((batch_size,), label, device=device)
                x_in = overlay_y_on_x(data, y_candidate, device=device)

                model.reset_state()
                for t in range(config["time_steps"]):
                    model(x_in, phase="test")

                g_stats = model.get_goodness(reduction="none")
                vals = [v for k, v in g_stats.items() if k.endswith("_goodness")]
                if len(vals) > 0:
                    total_g = torch.stack(vals).sum(dim=0)
                else:
                    total_g = torch.zeros(batch_size, device=device)
                goodness_matrix[:, label] = total_g

            preds = goodness_matrix.argmax(dim=1)
            correct += preds.eq(target).sum().item()
            total += batch_size

    return 100.0 * correct / total


def run_single_trial(trial_id, args, device, train_loader, test_loader):
    logger.info(f"--- Starting Trial {trial_id} ---")

    config = {
        "input_dim": 784,
        "hidden_dim": 1500,
        "num_layers": 2,
        "learning_rate": 0.005,
        "ff_threshold": 4.0,
        "tau_mem": 0.5,
        "threshold": 1.0,
        "dt": 1.0,
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs
    }

    model = VisualCortex(device=device, config=config).to(device)

    # Training
    for epoch in range(1, config["epochs"] + 1):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            x_pos = overlay_y_on_x(data, target, device=device)
            y_neg = (target + torch.randint(1, 10,
                     (batch_size,), device=device)) % 10
            x_neg = overlay_y_on_x(data, y_neg, device=device)

            # Positive
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_pos, phase="wake")

            # Negative
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_neg, phase="sleep")

    # Final Validation
    acc = validate(model, test_loader, device, config)
    logger.info(f"Trial {trial_id} Finished. Accuracy: {acc:.2f}%")
    return acc


def main():
    parser = argparse.ArgumentParser(description="DORA Stability Benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--time_steps", type=int, default=25)
    parser.add_argument("--threshold", type=float,
                        default=90.0, help="Success threshold accuracy %")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    logger.info(
        f"Running Stability Benchmark: {args.runs} runs, Target > {args.threshold}%")

    # Dataset Preparation (Once)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    train_ds = datasets.MNIST(data_dir, train=True,
                              download=True, transform=base_transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=base_transform)

    # Use subset for speed if needed, but for benchmark full set is better.
    # For speed in development, maybe limit? No, sticking to defaults.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=100,
                             shuffle=False, num_workers=0)

    accuracies = []

    for i in range(args.runs):
        acc = run_single_trial(i+1, args, device, train_loader, test_loader)
        accuracies.append(acc)

    accuracies = np.array(accuracies)
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    success_rate = np.mean(accuracies >= args.threshold) * 100.0

    logger.info("=== Benchmark Results ===")
    logger.info(f"Accuracies: {accuracies}")
    logger.info(f"Mean Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}")
    logger.info(
        f"Stability Score (Success Rate > {args.threshold}%): {success_rate:.1f}%")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": vars(args),
        "accuracies": accuracies.tolist(),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "stability_score": success_rate
    }

    with open("stability_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
