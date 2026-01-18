import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
import numpy as np
import argparse
import json
import logging
import time
from datetime import datetime
from tqdm import tqdm

# Path setup
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../')))

try:
    from snn_research.models.visual_cortex import VisualCortex
except ImportError:
    # Fallback if relative import fails, try adding root explicitly if needed
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../../')))
    from snn_research.models.visual_cortex import VisualCortex

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StabilityBenchmark")

# Progress File for Dashboard
PROGRESS_FILE = "workspace/runtime_state/benchmark_progress.json"


def save_progress(epoch, accuracy, status="RUNNING"):
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump({
                "timestamp": time.time(),
                "epoch": epoch,
                "accuracy": accuracy,
                "status": status
            }, f)
    except Exception as e:
        pass


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

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        batch_size = data.size(0)
        goodness_matrix = torch.zeros(batch_size, 10, device=device)

        for label in range(10):
            y_candidate = torch.full((batch_size,), label, device=device)
            x_in = overlay_y_on_x(data, y_candidate, device=device)

            model.reset_state()
            with torch.no_grad():
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

    if total == 0:
        return 0.0
    return 100.0 * correct / total


def run_single_trial(trial_id, args, device, train_loader, test_loader):
    logger.info(f"--- Starting Trial {trial_id} ---")

    # Tuned Configuration for Stability
    config = {
        "input_dim": 784,
        "hidden_dim": 1500,
        "num_layers": 2,
        # Tuned for Learning: Higher LR, Lower Threshold
        "learning_rate": 0.05,  # Increased from 0.002
        # Lower threshold to encourage activity 2.0 is standard for FF
        # Lower threshold to encourage activity 2.0 is standard for FF
        "ff_threshold": 2.0,    # Decreased from 5.0
        "tau_mem": 20.0,        # Fixed for stability (dt=1.0)
        "threshold": 1.0,
        "dt": 1.0,
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "use_layer_norm": True
    }

    model = VisualCortex(device=device, config=config).to(device)

    # Metrics history
    history = {
        "accuracy": [],
        "layer_stats": []
    }

    # Training
    for epoch in range(1, config["epochs"] + 1):
        model.train()

        # Track epoch-level stats
        epoch_firing_rates = []

        for data, target in tqdm(train_loader, desc=f"Trial {trial_id} Epoch {epoch}/{config['epochs']}", unit="batch"):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # Scale input to prevent immediate saturation
            x_pos = overlay_y_on_x(data, target, device=device)
            y_neg = (target + torch.randint(1, 10,
                     (batch_size,), device=device)) % 10
            x_neg = overlay_y_on_x(data, y_neg, device=device)

            # Positive
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_pos, phase="wake")

            # Debug: Capture firing rate after positive phase
            state_pos = model.get_state()
            epoch_firing_rates.append(state_pos["layers"]["V1"]["firing_rate"])

            # Negative
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_neg, phase="sleep")

        # Log average firing rate for V1
        avg_rate = sum(epoch_firing_rates) / \
            len(epoch_firing_rates) if epoch_firing_rates else 0.0

        print(f"DEBUG_FIRING_RATE: {avg_rate}")
        try:
            with open("workspace/diagnosis_report.json", "w") as f:
                json.dump({"last_firing_rate": avg_rate, "epoch": epoch}, f)
        except Exception:
            pass

        logger.info(
            f"  Epoch {epoch} Avg Firing Rate (V1): {avg_rate:.4f} (Target ~0.1-0.5)")
        logger.info(
            f"  Epoch {epoch} Avg Firing Rate (V1): {avg_rate:.4f} (Target ~0.1-0.5)")

        # Record detailed state occasionally
        # We can't log every batch, too heavy.

        # Validation per epoch (optional, but good for tracking)
        # acc = validate(model, test_loader, device, config)
        # logger.info(f"Epoch {epoch} Acc: {acc:.2f}%")

        # Log internal state sample
        state = model.get_state()
        history["layer_stats"].append(state["layers"])

        # Update Progress for Dashboard
        # Estimate accuracy or use validation per epoch
        # For speed transparency, we valid every epoch here if needed, or just log "Training"
        # Let's do a quick check on test subset or just log epoch completion
        save_progress(
            epoch, 0.0, status=f"Training Trial {trial_id} - Epoch {epoch}/{config['epochs']}")

    # Final Validation
    acc = validate(model, test_loader, device, config)
    logger.info(f"Trial {trial_id} Finished. Accuracy: {acc:.2f}%")

    save_progress(config["epochs"], acc, status=f"Trial {trial_id} Finished")

    # Check "State Collapse" (if V1 weights are all zero or massive)
    v1_stats = history["layer_stats"][-1]["V1"]
    logger.info(
        f"  V1 Final State: Mem={v1_stats['mean_mem']:.4f}, W_mean={v1_stats['weight_mean']:.4f}")

    return acc, history


def main():
    parser = argparse.ArgumentParser(description="DORA Stability Benchmark")
    parser.add_argument("--runs", type=int, default=5, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per trial")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--time_steps", type=int, default=25)
    parser.add_argument("--threshold", type=float,
                        default=95.0, help="Success threshold accuracy %")
    parser.add_argument("--subset_size", type=int, default=None,
                        help="Limit training dataset size for speed")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    logger.info(
        f"Running Stability Benchmark v2.1: {args.runs} runs, Target > {args.threshold}%")

    # Dataset Preparation (Once)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])
    data_dir = os.path.join(os.path.dirname(__file__), '../data')
    # Use subset for speed if basic test
    train_ds = datasets.MNIST(data_dir, train=True,
                              download=True, transform=base_transform)
    test_ds = datasets.MNIST(data_dir, train=False, transform=base_transform)

    if args.subset_size:
        logger.info(f"Using subset of size {args.subset_size} for speed.")
        indices = torch.randperm(len(train_ds))[:args.subset_size]
        train_ds = torch.utils.data.Subset(train_ds, indices)
        # Also limit test set to speed up validation?
        # Maybe proportionate? Let's keep test set full or small fixed
        test_indices = torch.randperm(len(test_ds))[:1000]
        test_ds = torch.utils.data.Subset(test_ds, test_indices)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=100,
                             shuffle=False, num_workers=0)

    accuracies = []

    for i in range(args.runs):
        acc, _ = run_single_trial(i+1, args, device, train_loader, test_loader)
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

    with open("workspace/benchmarks/stability_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
