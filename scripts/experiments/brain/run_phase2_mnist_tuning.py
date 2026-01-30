# path: scripts/experiments/brain/run_phase2_mnist_tuning.py
# run_phase2_mnist_tuning
# 目的: パラメータ完全固定による「大きさ」ではなく「向き」の学習 (Rev79)

import sys
import os
import logging
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from typing import Optional
import math

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from snn_research.models.visual_cortex_v2 import VisualCortexV2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', force=True)
logger = logging.getLogger("Phase2_MNIST_Rev79")

class MNISTOverlayProcessor:
    """MNIST画像プロセッサ (MinMax & Unit-Norm)"""
    def __init__(self, device):
        self.device = device
    
    def overlay_label(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        batch_size = x.size(0)
        
        # 1. Min-Max Scaling (-1.0 to 1.0)
        # Rev58の成功要因。外れ値が出にくく安定する。
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x = (x - x_min) / (x_max - x_min + 1e-6)
        x = (x - 0.5) * 2.0
        
        if labels is None:
            label_vec = torch.zeros(batch_size, 10, device=self.device)
        else:
            labels = labels.to(self.device)
            one_hot = F.one_hot(labels, num_classes=10).float()
            # ラベル強度 1.5 (画像と対等)
            label_vec = one_hot * 1.5
            
        # 2. 結合
        combined = torch.cat([x, label_vec], dim=1)
        
        # 3. Strict Unit-Norm
        combined = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return combined

def generate_mixed_negatives(targets, device):
    """50% Hard / 50% Random Negative Sampling"""
    confusion_map = {
        0: [6, 9, 8], 1: [7, 4], 2: [3, 8], 3: [8, 5, 2],
        4: [9, 7, 1], 5: [3, 6, 8], 6: [5, 0, 8], 7: [1, 9, 4],
        8: [3, 0, 9, 5, 2], 9: [4, 7, 0]
    }
    neg_targets = targets.clone()
    for i in range(len(targets)):
        if np.random.random() < 0.5:
            true_lbl = targets[i].item()
            candidates = confusion_map[true_lbl]
            neg_targets[i] = candidates[np.random.randint(len(candidates))]
        else:
            neg_lbl = np.random.randint(0, 10)
            while neg_lbl == targets[i].item():
                neg_lbl = np.random.randint(0, 10)
            neg_targets[i] = neg_lbl
    return neg_targets.to(device)

def weight_fixed_constraint(brain):
    """重みのノルムを「2.0」に完全固定"""
    target_norm = 2.0
    with torch.no_grad():
        for name, param in brain.named_parameters():
            if 'weight' in name:
                norm = param.data.norm(p=2, dim=-1, keepdim=True)
                # 常に一定の長さに保つ
                param.data.copy_(param.data * (target_norm / (norm + 1e-6)))

def evaluate(brain, test_loader, processor, device):
    """V3重視・対数評価"""
    brain.eval()
    correct = 0
    total = 0
    limit = 600 
    
    loader = DataLoader(test_loader.dataset, batch_size=1, shuffle=True)
    pred_counts = {i: 0 for i in range(10)}

    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            if i >= limit: break
            data = data.to(device)
            target = target.item()
            
            label_scores = []
            for lbl in range(10):
                brain.reset_state()
                x_in = processor.overlay_label(data, torch.tensor([lbl], device=device))
                brain(x_in, phase="inference")
                stats = brain.get_goodness()
                
                gs = []
                for k, v in stats.items():
                    if "goodness" in k:
                        val = v.item() if torch.is_tensor(v) else float(v)
                        # V3重視 (3.0倍)
                        w = 3.0 if "V3" in k else (1.5 if "V2" in k else 1.0)
                        gs.append(math.log(max(1e-9, val)) * w)
                
                label_scores.append(sum(gs) if gs else -1e10)
            
            pred = np.argmax(label_scores)
            pred_counts[pred] += 1
            if pred == target: correct += 1
            total += 1

    acc = 100.0 * correct / total
    logger.info(f"📊 Prediction Distribution: {pred_counts}")
    return acc

def run_tuning():
    logger.info("🔧 Starting Phase 2 MNIST Tuning (Rev79: Static Constraints)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "input_dim": 794,
        "hidden_dim": 2560,
        "num_layers": 3,
        "learning_rate": 0.0015,
        "ff_threshold": 2.0,   # 固定閾値 (低めに設定し、確実に超えさせる)
        "w_decay": 0.0,        # 重み固定なので減衰は不要
        "sparsity": 0.08
    }
    
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    epochs = 12
    base_lr = config["learning_rate"]

    for epoch in range(1, epochs + 1):
        current_lr = base_lr * (0.95 ** (epoch - 1))
        
        # 閾値は固定 (2.0)
        
        brain.train()
        total_pos_g, total_neg_g = 0.0, 0.0
        
        logger.info(f"--- Epoch {epoch} Start (LR: {current_lr:.6f} | Thres: {brain.ff_threshold:.1f} | Norm: 2.0) ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # Positive
            brain.reset_state()
            x_pos = processor.overlay_label(data, target)
            brain(x_pos, phase="wake")
            with torch.no_grad():
                g_pos = brain.get_goodness()
                total_pos_g += float(sum(v for k, v in g_pos.items() if "goodness" in k))
            
            # Negative (Mixed)
            brain.reset_state()
            rnd_targets = generate_mixed_negatives(target, device)
            x_neg = processor.overlay_label(data, rnd_targets)
            brain(x_neg, phase="sleep")
            with torch.no_grad():
                g_neg = brain.get_goodness()
                total_neg_g += float(sum(v for k, v in g_neg.items() if "goodness" in k))
            
            # 重み固定 (Norm=2.0)
            weight_fixed_constraint(brain)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_pos = total_pos_g / 100.0
                avg_neg = total_neg_g / 100.0
                logger.info(f"  Step {batch_idx}: Pos={avg_pos:.2f} | Neg={avg_neg:.2f} | Margin={avg_pos - avg_neg:.2f}")
                total_pos_g, total_neg_g = 0.0, 0.0

        acc = evaluate(brain, test_loader, processor, device)
        logger.info(f"✅ Epoch {epoch} Final Accuracy: {acc:.2f}%")

def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    data_path = "workspace/data"
    os.makedirs(data_path, exist_ok=True)
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))

if __name__ == "__main__":
    try:
        run_tuning()
    except Exception as e:
        logger.exception(f"Fatal error: {e}")