# directory: snn_research/models/experimental
# file: sara_engine.py
# title: SARA Engine v11.5 (Predictive-Vision Rust Core)
# description: 予測符号化(Predictive Coding)を導入。トップダウン予測とボトムアップ入力の差分（誤差）のみをスパイク処理することで、認識精度と適応能力を強化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import logging

# Rust Kernel Import Strategy
try:
    import dora_kernel
    RUST_KERNEL_AVAILABLE = True
except ImportError:
    RUST_KERNEL_AVAILABLE = False

from snn_research.core.neurons.lif_neuron import LIFNeuron
from snn_research.cognitive_architecture.physics_evaluator import PhysicsEvaluator

logger = logging.getLogger(__name__)

if RUST_KERNEL_AVAILABLE:
    logger.info("✅ Rust acceleration kernel (dora_kernel) loaded successfully.")
else:
    logger.warning("⚠️ Rust kernel not found. Running in pure Python/PyTorch mode.")

# ==========================================
# 1. Visual Cortex (Visual Encoder)
# ==========================================
class VisualEncoder(nn.Module):
    """
    視覚入力を処理するCNNエンコーダー。
    画像の空間的特徴（エッジ、形状）を抽出し、ベクトル化します。
    """
    def __init__(self, input_channels: int = 1, feature_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.flatten_dim = 64 * 7 * 7
        self.fc = nn.Linear(self.flatten_dim, feature_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, C, H, W)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==========================================
# 2. Spiking Recurrent Core (Hybrid Rust/Python)
# ==========================================
class SpikingRNNCell(nn.Module):
    """
    LIFニューロンを用いたリカレントセル。
    dora_kernelが利用可能な場合はRustでニューロン更新計算を行います。
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 入力重みとリカレント重み
        self.input_weights = nn.Linear(input_dim, hidden_dim)
        self.recurrent_weights = nn.Linear(hidden_dim, hidden_dim)
        
        self.neuron = LIFNeuron(features=hidden_dim)
        
        self.decay = 0.9
        self.threshold = 1.0

    def forward(self, input_tensor: torch.Tensor, prev_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prev_spikes, prev_mem = prev_state
        
        # 電流統合 (Batch, Hidden)
        current = self.input_weights(input_tensor) + self.recurrent_weights(prev_spikes)
        
        # ニューロン状態更新
        if RUST_KERNEL_AVAILABLE and not self.training:
            # Rust Kernel
            device = current.device
            current_np = current.detach().cpu().numpy()
            prev_mem_np = prev_mem.detach().cpu().numpy()
            
            spikes_np, new_mem_np = dora_kernel.lif_update_step(
                current_np, 
                prev_mem_np, 
                self.decay, 
                self.threshold
            )
            
            spike = torch.from_numpy(spikes_np).to(device)
            new_mem = torch.from_numpy(new_mem_np).to(device)
            
        else:
            # PyTorch Fallback
            new_mem = self.decay * prev_mem + current
            spike = self.surrogate_activation(new_mem - self.threshold)
            new_mem = new_mem - (spike.detach() * self.threshold)
        
        return spike, (spike, new_mem)

    @staticmethod
    def surrogate_activation(x):
        return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

# ==========================================
# 3. Memory & Meta-Cognition
# ==========================================
class SARAMemory(nn.Module):
    def __init__(self, capacity: int = 1000, state_dim: int = 128):
        super().__init__()
        self.capacity = capacity
        self.state_dim = state_dim
        self.register_buffer('memory_states', torch.zeros(capacity, state_dim))
        self.register_buffer('memory_values', torch.zeros(capacity, 1))
        self.register_buffer('write_pointer', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_full', torch.zeros(1, dtype=torch.bool))

    def store(self, state: torch.Tensor, value: float = 1.0):
        if state.dim() > 1: state = state.mean(dim=0)
        if state.shape[-1] != self.state_dim: return
        idx = self.write_pointer.item()
        self.memory_states[idx] = state.detach()
        self.memory_values[idx] = value
        self.write_pointer.fill_((idx + 1) % self.capacity)
        if (idx + 1) == self.capacity: self.is_full.fill_(True)

    def retrieve(self, query: torch.Tensor, k: int = 1):
        if query.dim() == 1: query = query.unsqueeze(0)
        if query.shape[-1] != self.state_dim:
             return torch.zeros(query.size(0), k, self.state_dim).to(query.device), torch.zeros(query.size(0), k).to(query.device)
        mem_norm = F.normalize(self.memory_states, p=2, dim=1)
        query_norm = F.normalize(query, p=2, dim=1)
        scores = torch.mm(query_norm, mem_norm.t())
        top_k_scores, indices = torch.topk(scores, k, dim=1)
        return self.memory_states[indices], top_k_scores

class MetaCognitiveMonitor(nn.Module):
    def __init__(self, history_size: int = 100):
        super().__init__()
        self.history_size = history_size
        self.register_buffer('error_history', torch.zeros(history_size))
        self.register_buffer('pointer', torch.zeros(1, dtype=torch.long))

    def update(self, current_error: float) -> float:
        self.error_history[self.pointer] = current_error
        self.pointer = (self.pointer + 1) % self.history_size
        short_term = self.error_history[-10:].mean()
        long_term = self.error_history.mean()
        return F.relu(short_term - long_term).item()

# ==========================================
# 4. SARA Engine Main v11.5 (Predictive Coding)
# ==========================================
class SARAEngine(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 action_dim: int,
                 config: Dict[str, Any] = None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.config = config or {}
        
        # --- 1. Visual System ---
        self.use_vision = self.config.get("use_vision", True)
        self.img_size = int(input_dim ** 0.5)
        if self.img_size * self.img_size != input_dim:
            self.use_vision = False
            
        if self.use_vision:
            self.visual_cortex = VisualEncoder(input_channels=1, feature_dim=hidden_dim)
            rnn_input_dim = hidden_dim + action_dim
        else:
            self.visual_cortex = None
            rnn_input_dim = input_dim + action_dim

        # --- 2. Cognitive Core ---
        self.perception_core = SpikingRNNCell(rnn_input_dim, hidden_dim)
        
        # --- 3. Heads ---
        # Top-down prediction head (Spike -> Expected Feature)
        self.feature_predictor = nn.Linear(hidden_dim, hidden_dim) 
        
        if self.use_vision:
            self.sensory_decoder = nn.Sequential(
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
        else:
            self.sensory_decoder = nn.Linear(hidden_dim, input_dim)

        self.action_generator = nn.Linear(hidden_dim, action_dim)
        
        # --- 4. Support ---
        self.meta_monitor = MetaCognitiveMonitor()
        self.physics_evaluator = PhysicsEvaluator(self.config.get("physics", {}))
        self.memory = SARAMemory(capacity=1000, state_dim=hidden_dim)
        
        self.plasticity_level = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.energy_reserve = nn.Parameter(torch.tensor(1.0), requires_grad=False)

    def forward(self, 
                sensory_input: torch.Tensor, 
                prev_action: torch.Tensor,
                prev_state: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        SARA v11.5 Predictive Coding Step:
        1. Top-Down Prediction: 前回の状態から現在の入力を予測
        2. Bottom-Up Error: 実際の入力と予測の差分（誤差）を計算
        3. State Update: 誤差信号をRNNコアに入力
        """
        batch_size = sensory_input.size(0)
        prev_spikes, _ = prev_state

        # --- 1. Visual Encoding (Bottom-Up) ---
        if self.use_vision:
            img_input = sensory_input.view(-1, 1, self.img_size, self.img_size)
            actual_features = self.visual_cortex(img_input) # (Batch, Hidden)
        else:
            actual_features = sensory_input

        # --- 2. Predictive Coding Logic ---
        # 前回のスパイクから、今回観測されるはずの特徴量を予測 (Top-Down)
        predicted_features = self.feature_predictor(prev_spikes)
        
        # 予測誤差 (Error Signal)
        # SARAは「世界とのズレ」を処理する
        prediction_error = actual_features - predicted_features
        
        # --- 3. Core Update with Error Signal ---
        # 入力には予測誤差を使用（＋前回の行動）
        combined_input = torch.cat([prediction_error, prev_action], dim=1)
        spike_out, next_state = self.perception_core(combined_input, prev_state)
        
        # --- 4. Reconstruction & Action ---
        # 外部への出力や可視化用に、デコーダーで感覚入力を再構成
        pred_sensory = self.sensory_decoder(spike_out)
        
        sensory_error = sensory_input - pred_sensory
        reconstruction_loss = torch.norm(sensory_error, dim=1).mean()
        
        # --- 5. Meta-Cognition & Homeostasis ---
        surprise_score = self.meta_monitor.update(reconstruction_loss.item())
        self.plasticity_level.data = torch.clamp(
            self.plasticity_level + 0.01 * (surprise_score - 0.5), 0.01, 1.0
        )
        
        physics_consistency = self.physics_evaluator.evaluate_state_consistency(
            spike_out.unsqueeze(1)
        )
        base_action = self.action_generator(spike_out)
        consistency_factor = physics_consistency.unsqueeze(1)
        scale = torch.where(
            consistency_factor < 0.5,
            consistency_factor,
            torch.ones_like(consistency_factor)
        )
        action = torch.tanh(base_action * scale)

        if surprise_score > 0.8:
            self.memory.store(spike_out, value=surprise_score)
            
        energy_cost = 0.001 * reconstruction_loss + 0.0001 * torch.norm(action)
        self.energy_reserve.data = torch.clamp(self.energy_reserve - energy_cost, 0.0, 1.0)

        return {
            "action": action,
            "next_state": next_state,
            "pred_sensory": pred_sensory,
            "sensory_error": sensory_error,
            "features": actual_features,
            "prediction_error": prediction_error, # Debug info
            "loss_components": {
                "prediction_error": reconstruction_loss,
                "physics_penalty": 1.0 - physics_consistency.mean(),
                "energy_cost": energy_cost
            },
            "meta_info": {
                "plasticity": self.plasticity_level.item(),
                "surprise": surprise_score,
                "energy": self.energy_reserve.item()
            }
        }

    def get_initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def reset_homeostasis(self):
        self.plasticity_level.data.fill_(0.1)
        self.energy_reserve.data.fill_(1.0)