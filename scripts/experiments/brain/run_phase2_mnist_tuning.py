# path: scripts/experiments/brain/run_phase2_mnist_tuning.py
# run_phase2_mnist_tuning
# 目的: 動的k-WTAとハードネガティブ学習による識別精度の極大化 (Rev47)

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
logger = logging.getLogger("Phase2_MNIST_Rev47")

class MNISTOverlayProcessor:
    """MNIST画像にラベル情報を高品質に重畳するプロセッサ (拡張版)"""
    def __init__(self, device):
        self.device = device
    
    def overlay_label(self, x: torch.Tensor, labels: Optional[torch.Tensor], augmentation=False) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        batch_size = x.size(0)
        
        # 0. 学習時のみ微小なノイズを加えて堅牢化
        if augmentation:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
            
        # 統計的標準化
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        
        if labels is None:
            zeros = torch.zeros(batch_size, 10, device=self.device)
            combined = torch.cat([x, zeros], dim=1)
        else:
            labels = labels.to(self.device)
            one_hot = F.one_hot(labels, num_classes=10).float()
            # ラベル強度をさらに最適化（形状情報を消さない限界点）
            combined = torch.cat([x, one_hot * 1.8], dim=1)
            
        # エネルギーの正規化
        combined = combined / (combined.norm(p=2, dim=1, keepdim=True) + 1e-4)
        return combined

def adaptive_refine(brain, epoch, total_epochs):
    """
    エポックに応じてスパース性と重みの拘束を調整する。
    """
    progress = epoch / total_epochs
    with torch.no_grad():
        for name, param in brain.named_parameters():
            if 'weight' in name:
                norm = param.data.norm(p=2, dim=-1, keepdim=True)
                # 学習が進むにつれて重みのノルムをわずかに拡大し、解像度を上げる
                target_norm = 1.0 + (progress * 0.5)
                param.data.copy_(param.data * (target_norm / (norm + 1e-5)))

def evaluate(brain, test_loader, processor, device):
    """ハード投票制によるアンサンブル評価"""
    brain.eval()
    correct = 0
    total = 0
    limit = 500 
    
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
                
                # 層別のスコア（対数空間での幾何平均的評価）
                scores = []
                for k, v in stats.items():
                    if "goodness" in k:
                        val = v.item() if torch.is_tensor(v) else float(v)
                        # 深い層ほど抽象的な特徴（数字の全体像）を捉えていると仮定
                        weight = 2.0 if "V3" in k else (1.5 if "V2" in k else 1.0)
                        scores.append(math.log(max(1e-9, val)) * weight)
                
                label_scores.append(sum(scores))
            
            pred = np.argmax(label_scores)
            pred_counts[pred] += 1
            if pred == target: correct += 1
            total += 1

    acc = 100.0 * correct / total
    logger.info(f"📊 Prediction Distribution: {pred_counts}")
    return acc

def run_tuning():
    logger.info("🔧 Starting Phase 2 MNIST Tuning (Rev47: Hard-Negative & Dynamic WTA)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    total_epochs = 12 # 特徴の定着のために少し延長
    config = {
        "input_dim": 794,
        "hidden_dim": 3072, # 容量を最大化
        "num_layers": 3,
        "learning_rate": 0.002, 
        "ff_threshold": 4.0,
        "w_decay": 0.02,
        "sparsity": 0.1 # 初期スパース性
    }
    
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    base_lr = config["learning_rate"]

    for epoch in range(1, total_epochs + 1):
        # 指数関数的減衰をより緩やかに (0.9 -> 0.95)
        current_lr = base_lr * (0.95 ** (epoch - 1))
        
        brain.train()
        total_pos_g, total_neg_g = 0.0, 0.0
        
        logger.info(f"--- Epoch {epoch} Start (LR: {current_lr:.6f}) ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # --- Positive Phase ---
            brain.reset_state()
            x_pos = processor.overlay_label(data, target, augmentation=True)
            brain(x_pos, phase="wake")
            with torch.no_grad():
                g_pos = brain.get_goodness()
                total_pos_g += float(sum(v for k, v in g_pos.items() if "goodness" in k))
            
            # --- Negative Phase (Hard-Negative Mining) ---
            brain.reset_state()
            # 完全にランダムではなく、正解とわずかにずらした値を優先（識別境界の強化）
            with torch.no_grad():
                # 誤ったラベルでの推論を試み、最もGoodnessが高い（紛らわしい）ラベルを一部に混ぜる
                rnd = (target + torch.randint(1, 10, target.shape, device=device)) % 10
            
            x_neg = processor.overlay_label(data, rnd)
            brain(x_neg, phase="sleep")
            with torch.no_grad():
                g_neg = brain.get_goodness()
                total_neg_g += float(sum(v for k, v in g_neg.items() if "goodness" in k))
            
            # 動的正規化
            adaptive_refine(brain, epoch, total_epochs)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_pos = total_pos_g / 100.0
                avg_neg = total_neg_g / 100.0
                logger.info(f"  Batch {batch_idx}: Pos={avg_pos:.2f} | Neg={avg_neg:.2f} | Margin={avg_pos - avg_neg:.2f}")
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