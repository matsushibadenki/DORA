# directory: tests/models/experimental
# file: train_sara_concept.py
# title: Train SARA with Concept Grounding (Time-Unfolded)
# description: MNIST画像を30ステップ連続提示し、SNNの内部状態を安定化させた上で概念学習を行う改良版スクリプト。

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
    BATCH_SIZE = 64
    EPOCHS = 2
    HIDDEN_DIM = 256
    ACTION_DIM = 10
    TIME_STEPS = 30 # 画像を見る時間（思考時間）
    
    # 3. Data Loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # 4. Model Initialization
    input_dim = 28 * 28 
    config = {
        "use_vision": False, 
        "noise_level": 0.1 # 探索ノイズ少し多め
    }

    model = SARAEngine(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        action_dim=ACTION_DIM,
        config=config
    ).to(device)
    
    # Init Weights
    nn.init.uniform_(model.perception_core.input_weights.weight, -0.1, 0.1)

    # Learner
    stdp = STDP(learning_rate=0.02) # Concept学習用の調整

    logger.info("Starting Time-Unfolded Concept Training...")
    start_time = time.time()
    
    # Trace buffers
    rnn_input_dim = input_dim + ACTION_DIM + HIDDEN_DIM # Input + Action + TopDown

    for epoch in range(EPOCHS):
        model.train()
        total_correct = 0
        total_samples = 0
        
        pre_trace_input = torch.zeros(BATCH_SIZE, rnn_input_dim, device=device)
        post_trace = torch.zeros(BATCH_SIZE, HIDDEN_DIM, device=device)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch_idx, (data, target) in enumerate(pbar):
            flat_input = data.view(BATCH_SIZE, -1).to(device)
            
            with torch.no_grad():
                state = model.get_initial_state(BATCH_SIZE, device)
                prev_action = torch.zeros(BATCH_SIZE, ACTION_DIM, device=device)
                
                # Input Encoding
                pixel_prob = torch.clamp(flat_input, 0.0, 1.0)
                input_spikes = (torch.rand_like(pixel_prob) < pixel_prob).float()
                
                # --- Time Step Loop (Thinking Process) ---
                concept_votes = torch.zeros(BATCH_SIZE, 10, device=device)
                
                for t in range(TIME_STEPS):
                    # SARA Forward
                    # 最初の10ステップは概念学習させず、状態を安定させる (Warm-up)
                    learn_now = (t > 10) 
                    target_concept = target if learn_now else None
                    
                    output = model(
                        flat_input, # Static image input
                        prev_action, 
                        state, 
                        concept_target=target_concept
                    )
                    
                    state = output["next_state"]
                    spike_out = state[0]
                    prev_action = output["action"] # 行動もループさせる
                    
                    # STDP Learning (Only if learning is active)
                    if learn_now:
                        # Top-down信号も入力に含まれるため、trace更新は少し複雑だが
                        # ここでは簡易的に Combined Input を再構築して近似
                        # (厳密にはmodel内部でcombined_inputを取得すべきだが、インタフェース上外から見えないため)
                        # 今回はSTDPによる重み更新はスキップし、ConceptMemoryの学習に集中する
                        pass

                    # Accumulate Concept Votes (Time integration of decision)
                    concept_votes += output["concept_logits"]

                # --- Batch Evaluation ---
                pred_concept = concept_votes.argmax(dim=1)
                correct = pred_concept.eq(target).sum().item()
                total_correct += correct
                total_samples += BATCH_SIZE
                
                pbar.set_postfix({
                    "Concept Acc": f"{100. * total_correct / total_samples:.2f}%"
                })

    logger.info(f"Training finished in {time.time() - start_time:.2f}s")
    logger.info(f"Final Concept Recall Accuracy: {100. * total_correct / total_samples:.2f}%")
    
    # 5. Imagination Test
    logger.info("Testing Imagination...")
    concept_id = torch.tensor([0], device=device) # "Zero"
    # 概念から想起されるプロトタイプベクトルを取得
    imagined_state = model.concept_memory.imagine_state(concept_id)
    # デコーダを通して画像空間へ射影
    imagined_image = model.sensory_decoder(imagined_state)
    logger.info(f"Imagined '0' intensity max: {imagined_image.max().item():.4f}")

if __name__ == '__main__':
    main()