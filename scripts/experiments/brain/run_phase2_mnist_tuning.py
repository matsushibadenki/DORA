# path: scripts/experiments/brain/run_phase2_mnist_tuning.py
# run_phase2_mnist_tuning
# 目的: 高コントラスト入力とZ-Score対数評価による精度80%への最終アプローチ (Rev64)

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
logger = logging.getLogger("Phase2_MNIST_Rev64")

class MNISTOverlayProcessor:
    """MNIST画像プロセッサ (高コントラスト・トポロジー強調版)"""
    def __init__(self, device):
        self.device = device
    
    def overlay_label(self, x: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
        x = x.view(x.size(0), -1).to(self.device)
        batch_size = x.size(0)
        
        # 1. 形状の強調: 標準化 + tanh によるエッジ増幅 (Rev58の成功をさらに強化)
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        x = torch.tanh(x * 1.5) # コントラストを強調
        
        # 2. ラベルの作成 (1.5 強度)
        if labels is None:
            label_vec = torch.zeros(batch_size, 10, device=self.device)
        else:
            labels = labels.to(self.device)
            label_vec = F.one_hot(labels, num_classes=10).float() * 1.5
            
        combined = torch.cat([x, label_vec], dim=1)
        # 3. ノルムを 2.0 に固定 (Rev62の安定性を継承)
        combined = combined * (2.0 / (combined.norm(p=2, dim=1, keepdim=True) + 1e-6))
        return combined

def weight_stabilization(brain, epoch):
    """重みの物理的拘束と解像度の段階的開放"""
    with torch.no_grad():
        target_norm = 1.0 + (epoch * 0.05) # Epochが進むごとに識別能力を拡張
        for name, param in brain.named_parameters():
            if 'weight' in name:
                norm = param.data.norm(p=2, dim=-1, keepdim=True)
                param.data.copy_(param.data * (target_norm / (norm + 1e-6)))

def evaluate_zlog(brain, test_loader, processor, device):
    """Z-Score突出度 + 対数階層加重評価 (最高精度のための最強ロジック)"""
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
            
            # 各ラベルにおける層別の Goodness を収集
            label_layer_gs = []
            for lbl in range(10):
                brain.reset_state()
                x_in = processor.overlay_label(data, torch.tensor([lbl], device=device))
                brain(x_in, phase="inference")
                stats = brain.get_goodness()
                l_gs = [v.item() if torch.is_tensor(v) else float(v) 
                        for k, v in stats.items() if "goodness" in k]
                label_layer_gs.append(l_gs)

            label_layer_gs = np.array(label_layer_gs) # [10, 3 layers]
            
            # 各層内でラベル間の Z-Score を計算 (突出度を抽出)
            z_scores = (label_layer_gs - label_layer_gs.mean(axis=0)) / (label_layer_gs.std(axis=0) + 1e-6)
            
            # 対数化と層別加重 (V3を最優先: 3.0)
            layer_weights = np.array([1.0, 1.5, 3.0])
            final_scores = (z_scores * layer_weights).sum(axis=1)
            
            pred = np.argmax(final_scores)
            pred_counts[pred] += 1
            if pred == target: correct += 1
            total += 1

    acc = 100.0 * correct / total
    logger.info(f"📊 Prediction Distribution (Z-Log): {pred_counts}")
    return acc

def run_tuning():
    logger.info("🔧 Starting Phase 2 MNIST Tuning (Rev64: Z-Log High Precision)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = {
        "input_dim": 794,
        "hidden_dim": 2560,
        "num_layers": 3,
        "learning_rate": 0.0012, 
        "ff_threshold": 3.5,
        "w_decay": 0.02,
        "sparsity": 0.08
    }
    
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=128)
    
    epochs = 12
    base_lr = config["learning_rate"]

    for epoch in range(1, epochs + 1):
        # 冷却スケジュールの調整
        current_lr = base_lr * (0.94 ** (epoch - 1))
        # Threshold の動的増加
        brain.ff_threshold = config["ff_threshold"] + (epoch * 0.15)
        
        brain.train()
        total_pos_g, total_neg_g = 0.0, 0.0
        
        logger.info(f"--- Epoch {epoch} Start (LR: {current_lr:.6f} | Thres: {brain.ff_threshold:.1f}) ---")
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # Positive Phase
            brain.reset_state()
            x_pos = processor.overlay_label(data, target)
            brain(x_pos, phase="wake")
            with torch.no_grad():
                g_pos = brain.get_goodness()
                total_pos_g += float(sum(v for k, v in g_pos.items() if "goodness" in k))
            
            # Negative Phase
            brain.reset_state()
            rnd = (target + torch.randint(1, 10, target.shape, device=device)) % 10
            x_neg = processor.overlay_label(data, rnd)
            brain(x_neg, phase="sleep")
            with torch.no_grad():
                g_neg = brain.get_goodness()
                total_neg_g += float(sum(v for k, v in g_neg.items() if "goodness" in k))
            
            # 重み安定化
            weight_stabilization(brain, epoch)
            
            if batch_idx % 100 == 0 and batch_idx > 0:
                avg_pos = total_pos_g / 100.0
                avg_neg = total_neg_g / 100.0
                logger.info(f"  Step {batch_idx}: Pos={avg_pos:.2f} | Neg={avg_neg:.2f} | Margin={avg_pos - avg_neg:.2f}")
                total_pos_g, total_neg_g = 0.0, 0.0

        # Z-Score 加重評価
        acc = evaluate_zlog(brain, test_loader, processor, device)
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