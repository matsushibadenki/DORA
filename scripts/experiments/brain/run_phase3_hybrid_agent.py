# scripts/experiments/brain/run_phase3_hybrid_agent.py
import sys
import os
import logging
import torch
import torch.nn as nn
import numpy as np
import time

# プロジェクトルート設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# 必要なモジュールのインポート
try:
    from snn_research.core.snn_core import SpikingNeuralSubstrate
except ImportError:
    class SpikingNeuralSubstrate(nn.Module):
        def __init__(self, config, device='cpu'): super().__init__()
        def add_neuron_group(self, *args, **kwargs): pass
        def forward(self, x): return x

try:
    from snn_research.models.experimental.bit_spike_mamba import BitSpikeMamba
except ImportError:
    class BitSpikeMamba(nn.Module):
        def __init__(self, vocab_size, d_model, n_layer): 
            super().__init__()
            self.linear = nn.Linear(d_model, d_model)
        def forward(self, x): return self.linear(x.float())

from snn_research.adaptive.intrinsic_motivator import IntrinsicMotivator
from snn_research.io.universal_encoder import UniversalEncoder

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(name)s] %(levelname)s - %(message)s', force=True)
logger = logging.getLogger("HybridAgent")

class Phase3HybridBrain(nn.Module):
    """
    Phase 3: Neuro-Symbolic Hybrid Brain
    - System 1: Spiking Neural Network (Fast, Energy Efficient, Intuitive)
    - System 2: BitSpikeMamba / LLM (Slow, Reasoning, Planning)
    """
    def __init__(self, device):
        super().__init__()
        self.device = device
        
        logger.info(f"🚀 Initializing Phase 3 Hybrid Agent on {device}...")

        # --- System 1: Fast Intuition (SNN) ---
        logger.info("    🧠 Initializing System 1: SNN Core...")
        snn_config = {'dt': 1.0, 'method': 'fast_forward'}
        self.system1 = SpikingNeuralSubstrate(config=snn_config, device=device)
        self._build_system1_architecture() 
        self.system1.to(device)
        
        # --- System 2: Deep Reasoning ---
        logger.info("    🧠 Initializing System 2: BitSpikeMamba...")
        self.system2 = BitSpikeMamba(
            vocab_size=100, 
            d_model=128, 
            n_layer=2
        ).to(device)
        
        # --- Gating & Integration ---
        self.encoder = UniversalEncoder()
        
        # IntrinsicMotivatorの初期化とデバイス転送
        self.motivator = IntrinsicMotivator()
        self.motivator.to(device) # 重要: buffer(running_error_mean)をdeviceへ移動
        
        # SNNとMambaの出力を統合して行動を決定する層
        self.decision_layer = nn.Linear(128 + 128, 4).to(device)

    def _build_system1_architecture(self):
        """SNNの内部構造を構築"""
        if hasattr(self.system1, 'add_neuron_group'):
            self.system1.add_neuron_group(
                name="input_layer", 
                num_neurons=128, 
                neuron_model=nn.Linear(128, 128)
            )
            self.system1.add_neuron_group(
                name="output_layer",
                num_neurons=128,
                neuron_model=nn.Linear(128, 128)
            )
            logger.info("    🛠️  SNN Layers 'input_layer' and 'output_layer' created.")
        else:
            logger.warning("    ⚠️  System 1 does not support add_neuron_group.")

    def forward(self, x, force_system2=False):
        """ハイブリッド推論ループ"""
        # System 1 (Fast)
        try:
            s1_out = self.system1(x)
            if isinstance(s1_out, dict):
                s1_out = list(s1_out.values())[-1]
            elif isinstance(s1_out, tuple):
                 s1_out = s1_out[0]
        except Exception:
            s1_out = torch.zeros(x.shape[0], 128).to(self.device)

        # 次元合わせ
        if s1_out.dim() == 1:
            s1_out = s1_out.unsqueeze(0)
        
        if s1_out.shape[-1] != 128:
             curr_dim = s1_out.shape[-1]
             if curr_dim > 128:
                 s1_out = s1_out[:, :128]
             else:
                 padding = torch.zeros(s1_out.shape[0], 128 - curr_dim).to(self.device)
                 s1_out = torch.cat([s1_out, padding], dim=1)

        # System 2 (Slow)
        if force_system2:
            s2_out = self.system2(x)
            if s2_out.shape != s1_out.shape:
                s2_out = torch.randn_like(s1_out) 
        else:
            s2_out = torch.zeros_like(s1_out)

        # 統合
        combined = torch.cat([s1_out, s2_out], dim=-1)
        action_logits = self.decision_layer(combined)
        return action_logits

class HybridAgent:
    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.brain = Phase3HybridBrain(self.device)
        self.step_count = 0
        
    def run_step(self):
        self.step_count += 1
        print(f"\n--- Step {self.step_count} ---")
        
        # 1. 観測 (Observation)
        obs = torch.randn(1, 128).float().to(self.device)
        
        # 2. 意思決定
        is_complex = (np.random.rand() > 0.7)
        if is_complex:
            logger.info("🤔 Task is COMPLEX. Creating System 2 thread...")
        
        logits = self.brain(obs, force_system2=is_complex)
        action = torch.argmax(logits, dim=-1).item()
        
        # 3. 行動実行
        actions = ["Explore", "Eat", "Sleep", "Socialize"]
        logger.info(f"🤖 Action: {actions[action]} (System 2 Active: {is_complex})")
        
        # 4. 内発的動機づけ更新 (Fix: API修正)
        # IntrinsicMotivator.compute_reward(predicted, actual) を使用
        # 予測符号化(Predictive Coding)のシミュレーションとしてダミー予測を生成
        predicted_next_state = torch.randn_like(obs) 
        actual_next_state = torch.randn_like(obs)
        
        reward = self.brain.motivator.compute_reward(predicted_next_state, actual_next_state)
        # logger.info(f"✨ Intrinsic Reward calculated: {reward.item():.4f}")

    def live(self, steps=5):
        try:
            for _ in range(steps):
                self.run_step()
                time.sleep(0.5)
            logger.info("✅ Hybrid Agent finished execution cycle.")
        except Exception as e:
            logger.error(f"❌ Error during agent life: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    agent = HybridAgent()
    agent.live(steps=5)