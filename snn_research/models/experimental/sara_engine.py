# directory: snn_research/models/experimental
# file: sara_engine.py
# title: SARA Engine v12.2 (Attractor Dynamics & Concept Loop)
# description: 概念想起からのトップダウン・フィードバックを実装し、アトラクターダイナミクスによる認識の安定化を図ったバージョン。imagine_stateの入力型対応を強化。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union
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
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==========================================
# 2. Spiking Recurrent Core
# ==========================================
class SpikingRNNCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_weights = nn.Linear(input_dim, hidden_dim)
        self.recurrent_weights = nn.Linear(hidden_dim, hidden_dim)
        self.neuron = LIFNeuron(features=hidden_dim)
        self.decay = 0.9
        self.threshold = 1.0

    def forward(self, 
                input_tensor: torch.Tensor, 
                prev_state: Tuple[torch.Tensor, torch.Tensor],
                noise_level: float = 0.0) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        prev_spikes, prev_mem = prev_state
        
        current = self.input_weights(input_tensor) + self.recurrent_weights(prev_spikes)
        
        if noise_level > 0.0 and self.training:
            noise = torch.randn_like(current) * noise_level
            current = current + noise

        if RUST_KERNEL_AVAILABLE and not self.training:
            device = current.device
            current_np = current.detach().cpu().numpy()
            prev_mem_np = prev_mem.detach().cpu().numpy()
            spikes_np, new_mem_np = dora_kernel.lif_update_step(
                current_np, prev_mem_np, self.decay, self.threshold
            )
            spike = torch.from_numpy(spikes_np).to(device)
            new_mem = torch.from_numpy(new_mem_np).to(device)
        else:
            new_mem = self.decay * prev_mem + current
            spike = self.surrogate_activation(new_mem - self.threshold)
            new_mem = new_mem - (spike.detach() * self.threshold)
        
        return spike, (spike, new_mem)

    @staticmethod
    def surrogate_activation(x):
        return (x > 0).float() - torch.sigmoid(x).detach() + torch.sigmoid(x)

# ==========================================
# 3. Concept Memory (Phase 2 Enhanced)
# ==========================================
class ConceptMemory(nn.Module):
    """
    SARA Phase 2: Concept Grounding Module.
    視覚状態と概念IDを双方向に結びつける連想メモリ。
    """
    def __init__(self, state_dim: int, num_concepts: int = 10):
        super().__init__()
        self.state_dim = state_dim
        self.num_concepts = num_concepts
        
        # Concept Prototypes (Attractors)
        self.register_buffer('concept_prototypes', torch.randn(num_concepts, state_dim))
        # 正規化初期化
        self.concept_prototypes = F.normalize(self.concept_prototypes, p=2, dim=1)
        
        self.register_buffer('concept_counts', torch.ones(num_concepts))

    def learn_concept(self, neural_state: torch.Tensor, concept_id: torch.Tensor, strength: float = 0.05):
        """
        Hebbian-like learning: Pull prototype closer to current neural state.
        """
        batch_size = neural_state.size(0)
        neural_state_norm = F.normalize(neural_state, p=2, dim=1)

        for i in range(batch_size):
            cid = concept_id[i].item()
            if cid < 0 or cid >= self.num_concepts:
                continue
                
            current_proto = self.concept_prototypes[cid]
            
            lr = strength / (1.0 + 0.001 * self.concept_counts[cid])
            delta = (neural_state_norm[i] - current_proto) * lr
            
            self.concept_prototypes[cid] += delta
            self.concept_counts[cid] += 1.0
            
        # Re-normalize prototypes to keep them on the hypersphere
        self.concept_prototypes = F.normalize(self.concept_prototypes, p=2, dim=1)

    def recall_concept(self, neural_state: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Bottom-up: State -> Concept
        Returns: Softmax probabilities or Logits
        """
        state_norm = F.normalize(neural_state, p=2, dim=1)
        # Cosine similarity: (Batch, NumConcepts)
        similarity = torch.mm(state_norm, self.concept_prototypes.t())
        return similarity / temperature

    def imagine_state(self, input_signal: torch.Tensor) -> torch.Tensor:
        """
        Top-down: Concept -> State (Attractor Signal)
        概念ID(int) または 概念ロジット(float) から、対応する神経状態を想像する。
        """
        # 入力が整数型(Long/Int)の場合はIDとして扱う
        if input_signal.dtype in [torch.long, torch.int, torch.int64, torch.int32]:
            # input_signal: (Batch,) of IDs
            return self.concept_prototypes[input_signal]
        
        # 入力が浮動小数点型(Float)の場合はロジット/確率として扱う
        else:
            # input_signal: (Batch, NumConcepts) of Logits
            # Ensure 2D
            if input_signal.dim() == 1:
                input_signal = input_signal.unsqueeze(0)
            
            attention = F.softmax(input_signal, dim=1)
            # (Batch, NumConcepts) @ (NumConcepts, StateDim) -> (Batch, StateDim)
            imagined_state = torch.mm(attention, self.concept_prototypes)
            return imagined_state

# ==========================================
# 4. Support Modules
# ==========================================
class SARAMemory(nn.Module):
    def __init__(self, capacity: int = 1000, state_dim: int = 128):
        super().__init__()
        self.capacity = capacity
        self.state_dim = state_dim
        self.register_buffer('memory_states', torch.zeros(capacity, state_dim))
        self.register_buffer('write_pointer', torch.zeros(1, dtype=torch.long))
        self.register_buffer('is_full', torch.zeros(1, dtype=torch.bool))

    def store(self, state: torch.Tensor, value: float = 1.0):
        if state.dim() > 1: state = state.mean(dim=0)
        idx = self.write_pointer.item()
        self.memory_states[idx] = state.detach()
        self.write_pointer.fill_((idx + 1) % self.capacity)
        if (idx + 1) == self.capacity: self.is_full.fill_(True)

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
# 5. SARA Engine Main v12.2
# ==========================================
class SARAEngine(nn.Module):
    """
    SARA Engine v12.2
    - Concept-State Loop (Attractor Dynamics)
    - Predictive Coding
    - Pure Spiking / Rust Core
    """
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
        
        # 1. Perception
        self.use_vision = self.config.get("use_vision", True)
        self.img_size = int(input_dim ** 0.5)
        
        if self.use_vision:
            self.visual_cortex = VisualEncoder(input_channels=1, feature_dim=hidden_dim)
            self.feature_dim = hidden_dim
        else:
            self.visual_cortex = None
            self.feature_dim = input_dim
            
        # Core Input: Prediction Error + Action + Top-down Concept
        rnn_input_dim = self.feature_dim + action_dim + hidden_dim

        # 2. Core
        self.perception_core = SpikingRNNCell(rnn_input_dim, hidden_dim)
        
        # 3. Heads
        self.feature_predictor = nn.Linear(hidden_dim, self.feature_dim) 
        self.sensory_decoder = nn.Linear(hidden_dim, input_dim) if not self.use_vision else \
                               nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())
        self.action_generator = nn.Linear(hidden_dim, action_dim)
        
        # 4. Cognition
        self.concept_memory = ConceptMemory(state_dim=hidden_dim, num_concepts=10)
        self.meta_monitor = MetaCognitiveMonitor()
        self.physics_evaluator = PhysicsEvaluator(self.config.get("physics", {}))
        self.memory = SARAMemory(capacity=1000, state_dim=hidden_dim)
        
        # Params
        self.plasticity_level = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.energy_reserve = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.base_noise_level = self.config.get("noise_level", 0.05)

    def forward(self, 
                sensory_input: torch.Tensor, 
                prev_action: torch.Tensor,
                prev_state: Tuple[torch.Tensor, torch.Tensor],
                concept_target: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        SARA Forward Step
        """
        prev_spikes, _ = prev_state

        # --- 1. Perception & Prediction ---
        if self.use_vision:
            img_input = sensory_input.view(-1, 1, self.img_size, self.img_size)
            actual_features = self.visual_cortex(img_input)
        else:
            actual_features = sensory_input

        predicted_features = self.feature_predictor(prev_spikes)
        prediction_error = actual_features - predicted_features
        
        # --- 2. Top-down Concept Feedback (Attractor) ---
        # 前回のスパイク状態から概念を想起し、その概念プロトタイプを入力として戻す
        with torch.no_grad():
            current_concept_logits = self.concept_memory.recall_concept(prev_spikes)
            top_down_signal = self.concept_memory.imagine_state(current_concept_logits) * 0.5 # Strength factor
        
        # --- 3. Core Update ---
        combined_input = torch.cat([prediction_error, prev_action, top_down_signal], dim=1)
        
        current_surprise = self.meta_monitor.error_history[-1].item()
        adaptive_noise = self.base_noise_level * (1.0 + current_surprise * 5.0)
        
        spike_out, next_state = self.perception_core(
            combined_input, 
            prev_state, 
            noise_level=adaptive_noise
        )
        
        # --- 4. Concept Learning ---
        # 新しい状態に基づいて概念想起を更新
        concept_logits = self.concept_memory.recall_concept(spike_out)
        
        if concept_target is not None and self.training:
            # 正解概念の方へプロトタイプを引き寄せる
            self.concept_memory.learn_concept(spike_out.detach(), concept_target)
        
        # --- 5. Action & Reconstruction ---
        pred_sensory = self.sensory_decoder(spike_out)
        base_action = self.action_generator(spike_out)
        action = torch.tanh(base_action) # Simplified for speed

        # Metrics
        sensory_error = sensory_input - pred_sensory
        reconstruction_loss = torch.norm(sensory_error, dim=1).mean()
        physics_consistency = self.physics_evaluator.evaluate_state_consistency(spike_out.unsqueeze(1))
        
        surprise_score = self.meta_monitor.update(reconstruction_loss.item())
        self.energy_reserve.data = torch.clamp(self.energy_reserve - 0.001, 0.0, 1.0)

        return {
            "action": action,
            "next_state": next_state,
            "pred_sensory": pred_sensory,
            "concept_logits": concept_logits,
            "loss_components": {
                "prediction_error": reconstruction_loss,
                "physics_penalty": 1.0 - physics_consistency.mean()
            },
            "meta_info": {
                "surprise": surprise_score,
                "noise": adaptive_noise
            }
        }

    def get_initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device))

    def reset_homeostasis(self):
        self.plasticity_level.data.fill_(0.1)
        self.energy_reserve.data.fill_(1.0)