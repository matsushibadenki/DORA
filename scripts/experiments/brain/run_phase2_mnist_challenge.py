# ファイルパス: scripts/experiments/brain/run_phase2_mnist_challenge.py
# 日本語タイトル: Phase 2 MNIST Challenge (Type Fixed)
# 目的: mypyエラー "Tensor not callable" を # type: ignore で抑制。

import sys
import time
import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
from typing import cast

sys.path.append(str(Path(__file__).resolve().parents[3]))

from app.containers import AppContainer
from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MNIST_Challenge")

class MNISTTrainer:
    def __init__(self, brain: ArtificialBrain, device: torch.device):
        self.brain = brain
        self.device = device
        
        # [Fix] Type ignore added
        self.brain.reset_state() # type: ignore
        with torch.no_grad():
            dummy_input = torch.zeros(1, 10).long().to(device)
            dummy_output = self.brain(dummy_input)
            
            if dummy_output.dim() > 2:
                input_dim = dummy_output.shape[-1]
            else:
                input_dim = dummy_output.shape[-1]
        
        # [Fix] Type ignore added
        self.brain.reset_state() # type: ignore
                
        logger.info(f"🧠 Detected Brain Output Dimension: {input_dim}")

        self.classifier_head = nn.Linear(input_dim, 10).to(device)
        
        self.optimizer = optim.Adam(
            list(self.brain.parameters()) + list(self.classifier_head.parameters()), 
            lr=5e-4, 
            weight_decay=1e-4
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        self.brain.train()
        self.classifier_head.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        self.brain.wake_up()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (data, target) in enumerate(pbar):
            # [Fix] Type ignore added
            self.brain.reset_state() # type: ignore
            
            data, target = data.to(self.device), target.to(self.device)
            input_tokens = (data.view(data.size(0), -1) * 255).long()
            input_tokens = torch.clamp(input_tokens, 0, 255)
            
            self.optimizer.zero_grad()
            
            features = self.brain(input_tokens)
            
            if features.dim() > 2:
                features = features.mean(dim=1)
                
            output = self.classifier_head(features)
            
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100. * correct / total:.2f}%"})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        logger.info(f"Epoch {epoch} Training Result: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        
        gc.collect()
        if self.device.type == 'mps':
            torch.mps.empty_cache()
            
        return avg_loss, accuracy

    def evaluate(self, test_loader: DataLoader):
        self.brain.eval()
        self.classifier_head.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                # [Fix] Type ignore added
                self.brain.reset_state() # type: ignore
                
                data, target = data.to(self.device), target.to(self.device)
                input_tokens = (data.view(data.size(0), -1) * 255).long()
                input_tokens = torch.clamp(input_tokens, 0, 255)
                
                features = self.brain(input_tokens)
                if features.dim() > 2:
                    features = features.mean(dim=1)
                
                output = self.classifier_head(features)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total
        logger.info(f"🧪 Evaluation Result: Loss={avg_loss:.4f}, Accuracy={accuracy:.2f}%")
        return avg_loss, accuracy

def run_mnist_challenge():
    print("\n" + "="*60)
    print("🔥 Artificial Brain Phase 2: MNIST Challenge (Memory Safe)")
    print("="*60 + "\n")

    container = AppContainer()
    
    config_path = Path("configs/templates/base_config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parents[3] / "configs/templates/base_config.yaml"
    container.config.from_yaml(str(config_path))
    
    container.config.device.from_value("cpu")
    
    brain = cast(ArtificialBrain, container.artificial_brain())
    device = brain.device
    
    print(f"✅ Brain Initialized on {device}")
    
    transform = transforms.Compose([
        transforms.Resize((14, 14)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_subset = torch.utils.data.Subset(train_dataset, range(5000))
    test_subset = torch.utils.data.Subset(test_dataset, range(1000))
    
    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    print(f"📚 Data Loaded: Train={len(train_subset)}, Test={len(test_subset)} (Resized to 14x14)")

    trainer = MNISTTrainer(brain, device)
    
    epochs = 5
    history = {"train_acc": [], "test_acc": []}
    
    print("\n🚀 Starting Training Loop...")
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = trainer.train_epoch(train_loader, epoch)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        
        brain.sleep_cycle()
        
    total_time = time.time() - start_time
    print(f"\n✨ Challenge Completed in {total_time:.1f}s")
    print(f"🏆 Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    
    if history['test_acc'][-1] > 80.0:
        print("🎉 SUCCESS: Brain successfully learned digit concepts!")
    elif history['test_acc'][-1] > 50.0:
        print("⚠️ PARTIAL SUCCESS: Learning observed, but optimization needed.")
    else:
        print("❌ FAILURE: Brain failed to generalize.")

if __name__ == "__main__":
    run_mnist_challenge()