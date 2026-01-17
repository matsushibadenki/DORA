import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm

# Update path BEFORE importing project modules
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../../../')))

# Verify path update
# print(f"DEBUG: sys.path augmented: {sys.path[-1]}")

try:
    from snn_research.models.visual_cortex import VisualCortex
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def flatten_tensor(x):
    return torch.flatten(x)


def overlay_y_on_x(x: torch.Tensor, y: torch.Tensor, num_classes: int = 10, device: str = "cpu"):
    """
    MNIST画像にOne-hotラベルを埋め込む（Supervised Forward-Forward用）
    """
    x_mod = x.clone()
    y_oh = F.one_hot(y, num_classes).float().to(device)
    scale_factor = 2.5

    # x is (Batch, 784)
    if x_mod.dim() == 2:
        x_mod[:, :num_classes] = y_oh * scale_factor
    return x_mod


def run_spiking_ff_mnist():
    print("=== Spiking Forward-Forward Experiment (True SNN) - Phase 2 Standard ===")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    # 1. Configuration
    config = {
        "input_dim": 784,
        "hidden_dim": 1500,
        "num_layers": 2,
        "learning_rate": 0.005,      # SNN Local Rule needs careful tuning
        "ff_threshold": 4.0,         # Target goodness
        "tau_mem": 0.5,
        "threshold": 1.0,
        "dt": 1.0,
        "time_steps": 25,            # Number of simulation steps per image
        "batch_size": 100,
        "epochs": 10
    }

    # 2. Dataset
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(flatten_tensor)
    ])

    data_dir = os.path.join(os.path.dirname(__file__), '../../../../data')
    # Download logic handled by datasets
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=base_transform)
    test_dataset = datasets.MNIST(
        data_dir, train=False, transform=base_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=100,
                             shuffle=False, num_workers=0)  # Smaller batch for test

    # 3. Model Initialization
    print("Initializing Visual Cortex...")
    model = VisualCortex(device=torch.device(device), config=config)
    model.to(device)  # Ensure module parameters are on device

    # 4. Training Loop
    print(f"Start SNN Training for {config['epochs']} epochs...")

    for epoch in range(1, config["epochs"] + 1):
        total_pos_goodness = 0.0
        total_neg_goodness = 0.0
        num_batches = 0

        # --- Training Phase ---
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for data, target in progress_bar:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)

            # A) Prepare Inputs
            # Positive: Image + Correct Label
            x_pos = overlay_y_on_x(data, target, device=device)

            # Negative: Image + Incorrect Label
            y_neg = (target + torch.randint(1, 10,
                     (batch_size,), device=device)) % 10
            x_neg = overlay_y_on_x(data, y_neg, device=device)

            # B) SNN Simulation (Time Loop)
            # 1. Positive Phase
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_pos, phase="wake")

            g_pos_final = model.get_goodness(reduction="mean")
            # Using first layer metric for logging
            pos_g_val = list(g_pos_final.values())[0]

            # 2. Negative Phase
            model.reset_state()
            for t in range(config["time_steps"]):
                model(x_neg, phase="sleep")

            g_neg_final = model.get_goodness(reduction="mean")
            neg_g_val = list(g_neg_final.values())[0]

            total_pos_goodness += pos_g_val
            total_neg_goodness += neg_g_val
            num_batches += 1

            progress_bar.set_postfix({
                "Pos": f"{pos_g_val:.2f}",
                "Neg": f"{neg_g_val:.2f}"
            })

        print(f"Epoch {epoch} Finished. Avg Pos: {total_pos_goodness/max(1, num_batches):.4f}, Avg Neg: {total_neg_goodness/max(1, num_batches):.4f}")

        # --- Validation Phase (Accuracy) ---
        if epoch % 1 == 0:
            print("Validating...")
            acc = validate(model, test_loader, device, config)
            print(f"Epoch {epoch} Accuracy: {acc:.2f}%")

    print("Experiment Finished.")
    save_dir = os.path.join(os.path.dirname(__file__),
                            '../../../../models/checkpoints')
    os.makedirs(save_dir, exist_ok=True)
    try:
        torch.save(model.state_dict(), os.path.join(
            save_dir, "visual_cortex_phase2.pth"))
    except Exception as e:
        print(f"Failed to save model: {e}")


def validate(model, loader, device, config):
    correct = 0
    total = 0

    limit_samples = 1000
    processed_samples = 0

    for data, target in tqdm(loader, desc="Validation", leave=False):
        if processed_samples >= limit_samples:
            break

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
        processed_samples += batch_size

    if total == 0:
        return 0.0

    return 100.0 * correct / total


if __name__ == "__main__":
    run_spiking_ff_mnist()
