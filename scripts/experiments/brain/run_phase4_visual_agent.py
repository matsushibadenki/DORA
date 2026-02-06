# scripts/experiments/brain/run_phase4_visual_agent.py
import sys
import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# --- Mock Imports (環境に依存せず動作するようにフォールバックを用意) ---
try:
    from snn_research.core.snn_core import SpikingNeuralSubstrate
except ImportError:
    class SpikingNeuralSubstrate(nn.Module):
        def __init__(self, config={}, device='cpu'): super().__init__()
        def add_neuron_group(self, *args, **kwargs): pass
        def forward(self, x): return torch.zeros(x.shape[0], 128).to(x.device)

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    class BitSpikeMamba(nn.Module):
        def __init__(self, **kwargs): super().__init__()
        def forward(self, x): return x

from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("VisualAgent")

class VisualTokenizer(nn.Module):
    """
    網膜（Retina）から視覚野（V1）へのエンコーディングを行うモジュール。
    画像(1x28x28)をVisual Tokens(Sequence of features)に変換。
    """
    def __init__(self, output_dim=128):
        super().__init__()
        # 簡易的なCNN: (B, 1, 28, 28) -> (B, output_dim)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, output_dim)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.flatten(start_dim=1)
        tokens = self.fc(x)
        return tokens # (B, 128)

class Phase4VisualBrain(nn.Module):
    """
    Phase 4: Visual Embodied Brain
    - Input: Image stream (MNIST sequence)
    - System 1: SNN (Visual Reflex / Fast Recognition)
    - System 2: Mamba (Visual Reasoning / Planning)
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        logger.info("    👁️ Initializing Visual Cortex (Tokenizer)...")
        self.visual_cortex = VisualTokenizer(output_dim=128).to(device)

        # --- System 1: Fast Intuition (SNN) ---
        logger.info("    🧠 Initializing System 1: SFormer (Visual Reflex)...")
        snn_config = {'dt': 1.0, 'method': 'fast_forward'}
        self.system1 = SpikingNeuralSubstrate(config=snn_config, device=device)
        self._build_snn_architecture() # 重要: 層を追加
        self.system1.to(device)

        # --- System 2: Deep Reasoning ---
        logger.info("    🧠 Initializing System 2: BitSpikeMamba (Visual Reasoning)...")
        self.system2 = BitSpikeMamba(vocab_size=10, d_model=128, n_layer=2).to(device)
        
        # --- Motivation & Action ---
        self.motivator = IntrinsicMotivator()
        self.motivator.to(device)
        
        # 行動出力: 0-9の数字予測、または探索行動
        self.motor_cortex = nn.Linear(128 * 2, 10).to(device)

    def _build_snn_architecture(self):
        """SNNの内部構造を構築 (IndexError回避)"""
        if hasattr(self.system1, 'add_neuron_group'):
            self.system1.add_neuron_group(
                name="input", 
                num_neurons=128, 
                neuron_model=nn.Linear(128, 128)
            )
            self.system1.add_neuron_group(
                name="output",
                num_neurons=128,
                neuron_model=nn.Linear(128, 128)
            )
            logger.info("       > SNN Layers initialized.")

    def forward(self, image, noise_level=0.0):
        # 1. Visual Encoding
        visual_tokens = self.visual_cortex(image) # (B, 128)
        
        if noise_level > 0:
            visual_tokens += torch.randn_like(visual_tokens) * noise_level

        # 2. System 1 (Fast)
        try:
            s1_out = self.system1(visual_tokens)
            if isinstance(s1_out, dict):
                s1_out = list(s1_out.values())[-1]
            elif isinstance(s1_out, tuple):
                 s1_out = s1_out[0]
        except Exception:
            s1_out = torch.zeros_like(visual_tokens)

        # 次元保証
        if s1_out.shape != visual_tokens.shape:
            s1_out = torch.zeros_like(visual_tokens)

        # 3. System 2 (Slow - Reasoning)
        # 視覚トークンをシーケンスとして扱う擬似処理
        s2_out = self.system2(visual_tokens)
        if s2_out.shape != s1_out.shape:
             s2_out = torch.randn_like(s1_out)

        # 4. Integration & Action
        combined_context = torch.cat([s1_out, s2_out], dim=1)
        action_logits = self.motor_cortex(combined_context)
        
        return {
            "action_logits": action_logits,
            "state_representation": combined_context,
            "visual_tokens": visual_tokens
        }

class VisualAgent:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"🚀 Initializing Phase 4 Visual Agent on {self.device}...")
        
        self.brain = Phase4VisualBrain(self.device)
        self.step = 0
        
        # データセット準備
        logger.info("    📥 Loading MNIST dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        # ダミーデータを使用（ダウンロード不要）
        self.dataset = [(torch.randn(1, 28, 28), torch.tensor(i % 10)) for i in range(10)]

    def run_life_cycle(self, steps=5):
        logger.info("🎬 Starting Visual Life Cycle...")
        
        for i in range(steps):
            self.step += 1
            print(f"\n--- Step {self.step} ---")
            
            # 環境からの入力 (画像)
            image, label = self.dataset[i % len(self.dataset)]
            image = image.unsqueeze(0).to(self.device) # (1, 1, 28, 28)
            
            # ノイズレベルを変動させる (環境の不確実性)
            noise = np.random.random() * 0.5
            
            # 脳の処理
            result = self.brain(image, noise_level=noise)
            logits = result["action_logits"]
            state = result["state_representation"]
            
            # 行動決定
            pred_class = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
            
            # 内発的報酬の計算 (予測誤差に基づく)
            # 次の状態を予測したと仮定 (Predictive Coding)
            predicted_next_state = state + 0.1 # ダミー予測
            actual_next_state = state # 実際には次のステップの観測を使うが、デモでは簡略化
            
            reward = self.brain.motivator.compute_reward(
                predicted_next_state, 
                actual_next_state
            )
            
            # ログ出力
            status = "✨ CONFIDENT" if confidence > 0.5 else "🤔 UNCERTAIN"
            logger.info(f"👁️ Visual Input: MNIST Class {label.item()} (Noise: {noise:.2f})")
            logger.info(f"🧠 Brain Output: Prediction {pred_class} | {status} ({confidence:.2f})")
            logger.info(f"🔥 Intrinsic Reward: {reward.item():.4f}")
            
            if confidence < 0.3:
                logger.info("   -> System 2 Triggered: 'I need to look closer...'")

        logger.info("✅ Phase 4 Visual Agent Cycle Completed.")

if __name__ == "__main__":
    agent = VisualAgent()
    agent.run_life_cycle(steps=5)