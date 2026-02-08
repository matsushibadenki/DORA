# directory: scripts/experiments/experimental
# file: train_sara_mnist.py
# purpose: SARA v5.0 Hybrid Training Script

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import os
from tqdm import tqdm

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    sys.path.append(".")
    from snn_research.models.experimental.sara_engine import SARAEngine

def generate_negative_samples(x):
    """
    ネガティブサンプルの生成
    バッチ内の画像をランダムにシャッフルし、強いノイズを加える
    """
    batch_size = x.size(0)
    # ランダムシャッフル（ラベルと画像の関係を破壊）
    idx = torch.randperm(batch_size)
    x_shuffled = x[idx]
    
    # ノイズ付加
    noise = torch.randn_like(x) * 0.5
    x_neg = x_shuffled + noise
    return x_neg

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    
    correct = 0
    total = 0
    running_ce_loss = 0.0
    running_ff_loss = 0.0
    avg_rate = 0.0
    
    # FFロスの重み（補助的な役割）
    ff_coeff = 0.5
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        x_pos = data.view(data.size(0), -1)
        x_neg = generate_negative_samples(x_pos).to(device)
        
        optimizer.zero_grad()
        
        # 1. Main Forward (Classification)
        logits, rate, _ = model(x_pos)
        ce_loss = F.cross_entropy(logits, target)
        
        # 2. Forward-Forward Loss (Auxiliary)
        ff_loss = model.compute_ff_loss(x_pos, x_neg)
        
        # Hybrid Loss
        loss = ce_loss + ff_coeff * ff_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Stats
        running_ce_loss += ce_loss.item()
        running_ff_loss += ff_loss.item()
        avg_rate = 0.9 * avg_rate + 0.1 * rate
        
        pred = logits.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)
        
        pbar.set_postfix({
            'CE': f"{ce_loss.item():.3f}", 
            'FF': f"{ff_loss.item():.3f}",
            'Acc': f"{100. * correct / total:.1f}%",
            'Rate': f"{avg_rate:.3f}"
        })

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            x = data.view(data.size(0), -1)
            
            logits, _, _ = model(x)
            test_loss += F.cross_entropy(logits, target, reduction='sum').item()
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Avg. loss: {test_loss:.4f}, Accuracy: {acc:.2f}%\n')

def main():
    BATCH_SIZE = 64
    EPOCHS = 5
    LR = 1e-3
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {DEVICE}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('../data', train=False, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # v5.0 Model
    model = SARAEngine(
        input_dim=784,
        n_encode_neurons=128,
        d_legendre=64,
        d_meaning=128,
        n_output=10
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(1, EPOCHS + 1):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
        
    torch.save(model.state_dict(), "models/checkpoints/sara_mnist_v5.pth")

if __name__ == "__main__":
    main()