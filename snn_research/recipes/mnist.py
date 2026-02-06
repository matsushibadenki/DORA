# snn_research/recipes/mnist.py
import os
import time
import logging
import random
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Any, Optional

# パッケージ内インポート
from snn_research.core.neurons import AdaptiveLIFNeuron
from snn_research.core.base import BaseModel

# 外部ライブラリ
try:
    from spikingjelly.activation_based import functional as SJ_F
    from spikingjelly.activation_based import neuron
except ImportError:
    pass

logger = logging.getLogger("Recipe_MNIST")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class MNIST_SOTA_SNN(BaseModel):
    """
    State-of-the-Art級の精度(>99.4%)を目指すSpiking CNNモデル。
    特徴: Deep Channel, Dropout, Parametric LIF
    """
    def __init__(self, num_classes=10, time_steps=4):
        super().__init__()
        self.time_steps = time_steps

        # Parametric LIF (学習可能な時定数を持つ)
        def create_neuron():
            return neuron.ParametricLIFNode(init_tau=2.0, detach_reset=True, surrogate_function=neuron.surrogate.ATan())

        # Encoder: 128chへの拡張とBatchNormによる安定化
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            create_neuron(),
            nn.MaxPool2d(2, 2), # 14x14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            create_neuron(),
            nn.MaxPool2d(2, 2), # 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256, bias=False),
            create_neuron(),
            nn.Dropout(0.5), # 過学習抑制
            nn.Linear(256, num_classes, bias=False)
        )

        self._init_weights()

    def forward(self, x):
        # Time-step展開: (B, C, H, W) -> (T, B, C, H, W)
        x_seq = x.unsqueeze(0).repeat(self.time_steps, 1, 1, 1, 1)
        
        # SpikingJelly Functional Reset
        SJ_F.reset_net(self)
        
        # Time-Distributed Forward
        outputs = []
        for t in range(self.time_steps):
            x_t = x_seq[t]
            feat = self.feature_extractor(x_t)
            out = self.classifier(feat)
            outputs.append(out)
        
        # Stack -> (T, B, Class) -> Mean over time -> (B, Class)
        outputs = torch.stack(outputs)
        return outputs.mean(dim=0)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True # 高速化

def run_mnist_training():
    parser = argparse.ArgumentParser(description="Train SOTA SNN on MNIST")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size") # 精度重視で少し小さめ
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # テストモード検知
    if os.environ.get("SNN_TEST_MODE") == "1":
        logger.info("🧪 Test Mode Detected: Reducing epochs and batch size")
        args.epochs = 1
        args.batch_size = 4

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Device: {device} | Epochs: {args.epochs}")

    # Data Augmentation (SOTAへの鍵)
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_path = "workspace/data"
    os.makedirs(data_path, exist_ok=True)
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = MNIST_SOTA_SNN(time_steps=4).to(device)
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine Annealing with Warm Restarts or simple Cosine
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_acc = 0.0
    
    for epoch in range(args.epochs):
        model.train()
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({"Acc": f"{100.*correct/total:.2f}%", "Loss": f"{loss.item():.4f}"})
        
        scheduler.step()

        # Validation
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
        
        acc = 100. * test_correct / test_total
        logger.info(f"Epoch {epoch+1}: Test Acc = {acc:.2f}% (Best: {max(best_acc, acc):.2f}%)")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "workspace/results/best_mnist_sota.pth")
            
            # 結果保存
            with open("workspace/results/best_mnist_metrics.json", "w") as f:
                json.dump({"accuracy": acc, "epoch": epoch+1}, f)

    logger.info(f"🏆 Final Best Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    run_mnist_training()