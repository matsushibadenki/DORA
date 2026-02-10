# directory: tests/models/experimental
# file: train_sara_r_stdp_mnist.py
# title: Pure SNN Training (R-STDP only)
# description: 誤差逆伝播法(Backprop)を一切使用せず、RustカーネルによるR-STDP（強化学習）のみで入力層から出力層までを学習する。

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', force=True)
logger = logging.getLogger(__name__)

from snn_research.models.experimental.sara_engine import SARAEngine
from snn_research.learning_rules.stdp import STDP

def main():
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (Optimized for Pure SNN / Rust Kernel)")

    # Hyperparameters
    BATCH_SIZE = 100
    EPOCHS = 1
    HIDDEN_DIM = 400
    ACTION_DIM = 10
    
    # Pure STDP Learning Rates
    LR_INPUT = 0.05    # Input -> Hidden (Unsupervised)
    LR_OUTPUT = 0.05   # Hidden -> Action (Reinforcement)

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Model
    config = {"use_vision": False} 
    model = SARAEngine(
        input_dim=28*28,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        config=config
    ).to(device)
    
    # Init Weights
    model.perception_core.input_weights.weight.data = torch.rand(HIDDEN_DIM, 28*28 + ACTION_DIM)
    
    # Action Generator も SNNの一部として扱うため、重みを正の値で初期化
    # (Hidden -> Action)
    model.action_generator.weight.data = torch.rand(ACTION_DIM, HIDDEN_DIM) * 0.1
    model.action_generator.bias.data.fill_(0.0)

    # Learners
    stdp_input = STDP(learning_rate=LR_INPUT)
    stdp_output = STDP(learning_rate=LR_OUTPUT)

    logger.info("Starting Pure R-STDP Training (No Backprop)...")
    start_time = time.time()
    
    total_correct = 0
    total_samples = 0
    
    # Traces
    rnn_input_dim = 28*28 + ACTION_DIM
    trace_input = torch.zeros(BATCH_SIZE, rnn_input_dim, device=device)
    trace_hidden = torch.zeros(BATCH_SIZE, HIDDEN_DIM, device=device)
    trace_action = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device) # Action用のトレース

    pbar = tqdm(train_loader)
    
    for batch_idx, (data, target) in enumerate(pbar):
        flat_input = data.view(BATCH_SIZE, -1)
        
        with torch.no_grad(): # 全工程勾配計算なし
            # 1. Encoding
            current_input = torch.clamp(flat_input, 0.0, 1.0)
            prev_action = torch.zeros(BATCH_SIZE, ACTION_DIM)
            combined_input = torch.cat([current_input, prev_action], dim=1)
            
            # 2. Hidden Layer (Perception Core)
            # Calculate Potential
            hidden_pot = torch.mm(combined_input, model.perception_core.input_weights.weight.t())
            
            # k-WTA (Sparse Activation)
            k = 3
            top_val, top_idx = torch.topk(hidden_pot, k, dim=1)
            hidden_spikes = torch.zeros_like(hidden_pot)
            hidden_spikes.scatter_(1, top_idx, 1.0)
            
            # 3. Action Layer (Output)
            # Simple Integrate: Hidden Spikes -> Action Potential
            action_pot = torch.mm(hidden_spikes, model.action_generator.weight.t())
            
            # Winner-Take-All for Action (Decide 1 class)
            action_idx = action_pot.argmax(dim=1)
            action_spikes = torch.zeros_like(action_pot)
            action_spikes.scatter_(1, action_idx.unsqueeze(1), 1.0)
            
            # 4. Reward Calculation (RLM)
            # 正解なら +1.0, 不正解なら -0.1 (罰)
            is_correct = action_idx.eq(target)
            reward_batch = torch.where(is_correct, torch.tensor(1.0), torch.tensor(-0.1))
            avg_reward = reward_batch.mean().item() # Batch平均報酬
            
            # 5. Update Traces
            trace_input = stdp_input.update_trace(trace_input, combined_input, stdp_input.tau_pre)
            trace_hidden = stdp_output.update_trace(trace_hidden, hidden_spikes, stdp_input.tau_post)
            # trace_action は Output学習用に HiddenをPre, ActionをPostとして使う

            # 6. Learning Step (R-STDP)
            
            # A. Hidden Layer Learning (Unsupervised / Self-Organizing)
            # 入力層->中間層は「特徴抽出」が目的なので、常にReward=1.0 (教師なし) で学習
            new_input_w = stdp_input.update(
                model.perception_core.input_weights.weight.data,
                combined_input,
                hidden_spikes,
                trace_input,
                trace_hidden,
                reward=1.0 
            )
            # Weight Norm
            norm = new_input_w.sum(dim=1, keepdim=True)
            new_input_w = new_input_w / (norm + 1e-6) * (28*28/2) 
            model.perception_core.input_weights.weight.data = new_input_w
            
            # B. Action Layer Learning (Reinforcement)
            # 中間層->出力層は「正解した時だけ強める」
            # Pre: Hidden Trace, Post: Action Spikes
            new_action_w = stdp_output.update(
                model.action_generator.weight.data,
                hidden_spikes,
                action_spikes,
                trace_hidden, # Pre trace (Hidden activity)
                action_spikes, # Post trace (Instant spike)
                reward=avg_reward # Global Reward Signal
            )
            model.action_generator.weight.data = new_action_w

            # Metrics
            correct = is_correct.sum().item()
            total_correct += correct
            total_samples += BATCH_SIZE
            
            pbar.set_postfix({
                "Acc": f"{100. * total_correct / total_samples:.2f}%",
                "Rew": f"{avg_reward:.2f}"
            })

    logger.info(f"Training finished in {time.time() - start_time:.2f}s")
    logger.info(f"Final Pure SNN Accuracy: {100. * total_correct / total_samples:.2f}%")

if __name__ == '__main__':
    main()