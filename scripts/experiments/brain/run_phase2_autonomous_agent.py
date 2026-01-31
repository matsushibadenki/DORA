# ファイルパス: scripts/experiments/brain/run_phase2_mnist_tuning.py
import sys
import os
import time
import logging
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from snn_research.models.visual_cortex import VisualCortex as VisualCortexV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("Phase2_MNIST_Rev15")

class MNISTOverlayProcessor:
    def __init__(self, device):
        self.device = device
    
    def overlay_label(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        batch_size = x.size(0)
        
        if labels is None:
            zeros = torch.zeros(batch_size, 10, device=self.device)
            return torch.cat([x, zeros], dim=1)
        else:
            labels = labels.to(self.device)
            one_hot = F.one_hot(labels, num_classes=10).float() 
            return torch.cat([x, one_hot], dim=1)

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    data_path = "workspace/data"
    os.makedirs(data_path, exist_ok=True)
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def evaluate(brain, test_loader, processor, device):
    brain.eval()
    correct = 0
    total = 0
    limit = 200 
    loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=True)
    
    pred_counts = {i: 0 for i in range(10)}
    debug_limit = 5

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if i >= limit: break
            data = data.to(device)
            target = target.item()
            
            scores = []
            
            for lbl in range(10):
                brain.reset_state()
                x_in = processor.overlay_label(data, torch.tensor([lbl], device=device))
                brain(x_in, phase="inference")
                stats = brain.get_goodness()
                raw = (stats.get("V1_goodness", 0) + 
                       stats.get("V2_goodness", 0) + 
                       stats.get("V3_goodness", 0))
                scores.append(raw)
            
            pred = np.argmax(scores)
            pred_counts[pred] += 1
            
            if pred == target: correct += 1
            total += 1
            
            if i < debug_limit:
                logger.info(f"Sample {i}: True={target} Pred={pred} | Scores={np.round(scores, 1).tolist()}")

    logger.info(f"Prediction Distribution: {pred_counts}")
    return 100.0 * correct / total

def run_tuning():
    logger.info("🔧 Starting Phase 2 MNIST Tuning (Rev15: Vitality Injection)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "input_dim": 794, 
        "hidden_dim": 2000,
        "num_layers": 3,
        "learning_rate": 0.08,
        "ff_threshold": 2000.0, 
        "w_decay": 0.0 # No decay to prevent death
    }
    
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=50)
    
    epochs = 3
    for epoch in range(1, epochs + 1):
        brain.train()
        total_pos_g = 0
        total_neg_g = 0
        batch_count = 0
        logger.info(f"--- Epoch {epoch} Start ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # Pos
            brain.reset_state()
            x_pos = processor.overlay_label(data, target)
            brain(x_pos, phase="wake")
            with torch.no_grad():
                total_pos_g += brain.get_goodness().get("V3_goodness", 0)
            
            # Neg
            brain.reset_state()
            rnd = torch.randint(0, 10, target.shape, device=device)
            rnd = torch.where(rnd == target, (rnd + 1) % 10, rnd)
            x_neg = processor.overlay_label(data, rnd)
            brain(x_neg, phase="sleep")
            with torch.no_grad():
                total_neg_g += brain.get_goodness().get("V3_goodness", 0)
            
            batch_count += 1
            if batch_count % 20 == 0:
                avg_pos = total_pos_g / 20
                avg_neg = total_neg_g / 20
                logger.info(f"Step {batch_idx}: Pos={avg_pos:.2f} | Neg={avg_neg:.2f} | Margin={avg_pos - avg_neg:.2f}")
                total_pos_g = 0
                total_neg_g = 0

        acc = evaluate(brain, test_loader, processor, device)
        logger.info(f"📊 Epoch {epoch} Accuracy: {acc:.1f}%")
        with open("workspace/reports/mnist_tuning_result.txt", "w") as f: f.write(f"Accuracy: {acc}%\n")

if __name__ == "__main__":
    run_tuning()