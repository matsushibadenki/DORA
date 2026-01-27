# path: scripts/experiments/brain/run_phase2_mnist_tuning.py
# run_phase2_mnist_tuning
# 目的: Rev46の識別力と絶対ノルム固定による安定化 (Rev54)

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
logger = logging.getLogger("Phase2_MNIST_Rev54")

class MNISTOverlayProcessor:
    """MNIST画像にラベル情報を重畳するプロセッサ (Rev46成功ロジック)"""
    def __init__(self, device):
        self.device = device
    
    def overlay_label(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        batch_size = x.size(0)
        
        # 1. 形状を際立たせる標準化 (成功の鍵)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        
        if labels is None:
            label_vec = torch.zeros(batch_size, 10, device=self.device)
        else:
            labels = labels.to(self.device)
            label_vec = F.one_hot(labels, num_classes=10).float() * 1.5 # 識別力が最も高かった強度
            
        # 2. 結合
        combined = torch.cat([x, label_vec], dim=1)
        
        # 3. エネルギーを1に固定 (Rev52の安定化)
        combined = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-6)
        return combined

def physical_stabilization(brain):
    """重みのノルムを1.0に物理的に固定する"""
    with torch.no_grad():
        for name, param in brain.named_parameters():
            if 'weight' in name:
                norm = param.data.norm(p=2, dim=-1, keepdim=True)
                param.data.copy_(param.data / (norm + 1e-6))

def evaluate(brain, test_loader, processor, device):
    """層別確信度の幾何平均評価"""
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
                
                # 層ごとのGoodnessを抽出し、対数空間で評価
                gs = []
                for k, v in stats.items():
                    if "goodness" in k:
                        val = v.item() if torch.is_tensor(v) else float(v)
                        # V3(概念)を1.5倍、他を1.0倍として合意形成
                        w = 1.5 if "V3" in k else 1.0
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
    logger.info("🔧 Starting Phase 2 MNIST Tuning (Rev54: Strict Physical Stable)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "input_dim": 794,
        "hidden_dim": 2048, # 安定のためやや絞る
        "num_layers": 3,
        "learning_rate": 0.001, # 低速・確実な学習
        "ff_threshold": 3.0,
        "w_decay": 0.05,
        "sparsity": 0.1
    }
    
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    epochs = 10
    base_lr = config["learning_rate"]

    for epoch in range(1, epochs + 1):
        # 指数減衰 (0.9)
        current_lr = base_lr * (0.9 ** (epoch - 1))
        
        brain.train()
        total_pos_g, total_neg_g = 0.0, 0.0
        
        logger.info(f"--- Epoch {epoch} Start (LR: {current_lr:.6f}) ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # --- Positive Phase ---
            brain.reset_state()
            x_pos = processor.overlay_label(data, target)
            brain(x_pos, phase="wake")
            with torch.no_grad():
                g_pos = brain.get_goodness()
                total_pos_g += float(sum(v for k, v in g_pos.items() if "goodness" in k))
            
            # --- Negative Phase ---
            brain.reset_state()
            # 難読ラベルによるコントラスト学習
            rnd = (target + torch.randint(1, 10, target.shape, device=device)) % 10
            x_neg = processor.overlay_label(data, rnd)
            brain(x_neg, phase="sleep")
            with torch.no_grad():
                g_neg = brain.get_goodness()
                total_neg_g += float(sum(v for k, v in g_neg.items() if "goodness" in k))
            
            # バッチごとの物理的強制正規化 (Rev54の肝)
            physical_stabilization(brain)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_pos = total_pos_g / 100.0
                avg_neg = total_neg_g / 100.0
                logger.info(f"  Batch {batch_idx}: Pos={avg_pos:.2f} | Neg={avg_neg:.2f} | Margin={avg_pos - avg_neg:.2f}")
                total_pos_g, total_neg_g = 0.0, 0.0

        acc = evaluate(brain, test_loader, processor, device)
        logger.info(f"✅ Epoch {epoch} Accuracy: {acc:.2f}%")

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