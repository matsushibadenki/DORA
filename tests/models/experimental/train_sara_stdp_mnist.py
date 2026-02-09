# directory: tests/models/experimental
# file: train_sara_stdp_mnist.py
# title: Train SARA with Rust STDP (Aggressive Homeostasis)
# description: 重み正規化と高学習率を導入し、1エポックでの強制学習を目指す最終調整版。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from tqdm import tqdm
import time

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

from snn_research.models.experimental.sara_engine import SARAEngine
from snn_research.learning_rules.stdp import STDP

def main():
    # 1. Device Configuration
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (Optimized for Rust Kernel)")

    # 2. Hyperparameters
    BATCH_SIZE = 100
    EPOCHS = 1
    HIDDEN_DIM = 400
    ACTION_DIM = 10
    
    # Parameters Boosted
    LEARNING_RATE_STDP = 0.1 # 爆上げ
    LEARNING_RATE_READOUT = 0.1

    # 3. Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    # 4. Model Initialization
    input_dim = 28 * 28 
    config = {"use_vision": False} 

    model = SARAEngine(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        config=config
    ).to(device)
    
    # Initialization
    # 平均0.5の乱数で初期化
    model.perception_core.input_weights.weight.data = torch.rand(HIDDEN_DIM, input_dim + ACTION_DIM)

    # 5. STDP Learner
    stdp = STDP(
        learning_rate=LEARNING_RATE_STDP,
        w_min=0.0,
        w_max=1.0,
        tau_pre=20.0,
        tau_post=20.0
    )

    readout_optimizer = torch.optim.SGD(model.action_generator.parameters(), lr=LEARNING_RATE_READOUT)

    # 6. Training Loop
    logger.info("Starting Hybrid Training (Aggressive STDP + Normalization)...")
    
    start_time = time.time()
    rnn_input_dim = input_dim + ACTION_DIM
    
    for epoch in range(EPOCHS):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pre_trace_input = torch.zeros(BATCH_SIZE, rnn_input_dim, device=device)
        post_trace = torch.zeros(BATCH_SIZE, HIDDEN_DIM, device=device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            flat_input = data.view(BATCH_SIZE, -1).to(device)
            target = target.to(device)

            # --- A. Unsupervised Learning (STDP) ---
            with torch.no_grad():
                prev_action = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                
                # Input: Direct current (0.0 - 1.0)
                # 画素値をそのまま入力強度として扱う
                current_input = torch.clamp(flat_input, 0.0, 1.0)
                combined_input = torch.cat([current_input, prev_action], dim=1)
                
                # Trace Update (Input activity itself acts as trace in rate-based approx)
                pre_trace_input = 0.5 * pre_trace_input + 0.5 * combined_input
                
                # --- SARA Core with k-WTA ---
                # Calculate Potential
                weights = model.perception_core.input_weights.weight
                potential = torch.mm(combined_input, weights.t())
                
                # Winner-Take-All (Top-1!)
                # 最も強く反応した1個だけを発火させる (Hard Competition)
                # これにより、各ニューロンが「特定の数字のプロトタイプ」にならざるを得なくなる
                k = 1 
                top_val, top_idx = torch.topk(potential, k, dim=1)
                
                spike_out = torch.zeros_like(potential)
                spike_out.scatter_(1, top_idx, 1.0)
                
                # Trace Update (Post)
                post_trace = 0.5 * post_trace + 0.5 * spike_out
                
                # Apply STDP (Rust)
                # Rustカーネルは「前後の相関」を見るので、Rate-basedでも機能する
                new_input_w = stdp.update(
                    weights,
                    combined_input, # spikes代わり
                    spike_out,
                    pre_trace_input,
                    post_trace
                )
                
                # --- Weight Normalization (Critical!) ---
                # 各ニューロンへの入力重みの合計を一定に保つ
                # これがないとSTDPは発散する
                norm_factor = new_input_w.sum(dim=1, keepdim=True)
                new_input_w = new_input_w / (norm_factor + 1e-6) * 784.0 * 0.5 # Target sum per neuron
                
                model.perception_core.input_weights.weight.data = new_input_w

            # --- B. Supervised Learning (Readout) ---
            readout_input = spike_out.detach()
            
            readout_optimizer.zero_grad()
            action_pred = model.action_generator(readout_input)
            loss = nn.functional.cross_entropy(action_pred, target)
            loss.backward()
            readout_optimizer.step()
            
            # --- Metrics ---
            pred = action_pred.argmax(dim=1)
            correct = pred.eq(target).sum().item()
            total_correct += correct
            total_samples += BATCH_SIZE
            
            pbar.set_postfix({
                "Acc": f"{100. * total_correct / total_samples:.2f}%",
            })

    total_time = time.time() - start_time
    logger.info(f"Training finished in {total_time:.2f}s")
    logger.info(f"Final Accuracy: {100. * total_correct / total_samples:.2f}%")

if __name__ == '__main__':
    main()