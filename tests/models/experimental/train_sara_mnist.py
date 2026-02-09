# directory: tests/models/experimental
# file: train_sara_mnist.py
# title: Train SARA Engine on MNIST
# description: ログ設定に force=True を追加し、Rustカーネルの読み込みメッセージを確実に表示するように修正。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from tqdm import tqdm

# --- 修正: force=True を追加して既存の設定を上書き ---
# これにより、PyTorch等が勝手に行ったログ設定をリセットし、INFOログを表示させます。
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)
# -------------------------------------------------------

from snn_research.models.experimental.sara_engine import SARAEngine

def main():
    # 1. Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    logger.info(f"Using device: {device}")

    # 2. Hyperparameters
    BATCH_SIZE = 64
    EPOCHS = 3
    HIDDEN_DIM = 128
    ACTION_DIM = 10 
    LEARNING_RATE = 1e-3

    # 3. Data Loading (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Model Initialization
    input_dim = 28 * 28 
    
    config = {
        "use_vision": True,
        "physics": {
            "smoothness_weight": 0.1
        }
    }

    model = SARAEngine(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        config=config
    ).to(device)

    # 5. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    logger.info("Starting training...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            flat_input = data.view(batch_size, -1)
            
            state = model.get_initial_state(batch_size, device)
            prev_action = torch.zeros(batch_size, ACTION_DIM, device=device)
            
            optimizer.zero_grad()
            
            output = model(flat_input, prev_action, state)
            action = output["action"]
            
            class_loss = nn.functional.cross_entropy(action * 5.0, target)
            internal_loss = sum(output["loss_components"].values())
            
            loss = class_loss + 0.1 * internal_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = action.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += batch_size
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Acc": f"{100. * correct / total:.2f}%",
                "Surprise": f"{output['meta_info']['surprise']:.4f}"
            })

    # 7. Evaluation
    logger.info("Evaluating...")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            flat_input = data.view(batch_size, -1)
            state = model.get_initial_state(batch_size, device)
            prev_action = torch.zeros(batch_size, ACTION_DIM, device=device)
            
            output = model(flat_input, prev_action, state)
            action = output["action"]
            
            test_loss += nn.functional.cross_entropy(action * 5.0, target).item()
            pred = action.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)
    logger.info(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)")

if __name__ == '__main__':
    main()