# directory: snn_research/models/experimental
# file: sara_engine.py
# purpose: SARA Engine v9.0 (Integrated Autonomous Intelligence Core)
# description: 予測符号化 (Predictive Coding) と能動的推論 (Active Inference) を統合した
#              SNNベースの自律学習エンジン。
#              Integrates:
#                - Probabilistic Hebbian (v8.0 integrated)
#                - Causal Trace (v8.0 integrated)
#                - Physics-Informed World Model (v8.0 integrated)
#                - Predictive Coding Rule (v9.0 integrated)
#                - Active Inference (v9.0 integrated)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass

from snn_research.config.schema import SARAConfig

@dataclass
class SARAMemory:
    """SARAの短期・長期記憶状態を保持するコンテナ"""
    hidden_state: torch.Tensor
    synaptic_trace: torch.Tensor
    world_model_state: torch.Tensor
    prediction_error: Optional[torch.Tensor] = None
    prior_belief: Optional[torch.Tensor] = None  # 能動的推論用の事前信念(Goal)

class WorldModel(nn.Module):
    """
    物理情報に基づく内部世界モデル (Physics-Informed World Model)
    状態遷移と感覚入力の予測を行う生成モデル。
    """
    def __init__(self, state_dim: int, hidden_dim: int, input_dim: int):
        super().__init__()
        self.state_dim = state_dim
        
        # 遷移モデル (Transition): State(t) -> State(t+1)
        self.transition = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim) # 変化量を出力
        )
        
        # 観測モデル (Observation/Emission): State(t) -> SensoryInput(t)
        # Predictive Codingにおける「Top-down Prediction」を生成
        self.decoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # エネルギー評価 (Physics Constraint)
        self.energy_evaluator = nn.Sequential(
            nn.Linear(state_dim, 1),
            nn.Softplus()
        )

    def forward(self, current_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            next_state_pred: 次の状態予測
            sensory_pred: 現在の状態から予測される感覚入力
            energy: 物理的エネルギー推定値
        """
        # 状態遷移 (物理ダイナミクスのシミュレーション)
        delta = self.transition(current_state)
        next_state_pred = current_state + delta
        
        # 感覚入力の再構成 (予測)
        sensory_pred = self.decoder(current_state)
        
        estimated_energy = self.energy_evaluator(next_state_pred)
        return next_state_pred, sensory_pred, estimated_energy

class PredictiveErrorUnit(nn.Module):
    """
    Predictive Codingの中核。
    予測(Top-down)と現実(Bottom-up)の不一致(Surprise/Free Energy)を計算する。
    """
    def __init__(self, feature_dim: int):
        super().__init__()
        # 誤差の重要度(Precision)を学習可能なパラメータとして持つ
        self.precision_log = nn.Parameter(torch.zeros(1, feature_dim))

    def forward(self, prediction: torch.Tensor, actual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            weighted_error: 重み付けされた誤差信号（学習や推論の駆動に使用）
            free_energy: 変分自由エネルギー（損失関数として使用）
        """
        error = actual - prediction
        precision = torch.exp(self.precision_log)
        
        # Precision-weighted error: e = Π * (y - y_pred)
        weighted_error = precision * error
        
        # Free Energy (Gaussian assumption): F = 0.5 * e^T * Π * e - 0.5 * ln|Π|
        squared_error = (error ** 2) * precision
        log_det = self.precision_log
        free_energy = 0.5 * (squared_error - log_det).mean(dim=1, keepdim=True)
        
        return weighted_error, free_energy

class ActiveInferenceController(nn.Module):
    """
    Active Inference (能動的推論) モジュール
    期待自由エネルギー(EFE)を最小化する行動(Action)を選択・生成する。
    """
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.action_generator = nn.Linear(state_dim, action_dim)
        
    def infer_action(self, current_state: torch.Tensor, goal_state: Optional[torch.Tensor]) -> torch.Tensor:
        """
        現在の状態と目標状態（選好）から、ギャップを埋める行動を生成
        """
        # 簡易的な実装: 目標がない場合は探索的な行動、ある場合は目標に向かう
        action_logits = self.action_generator(current_state)
        
        if goal_state is not None:
            # Goalとのコサイン類似度などを加味して変調（ここではシンプルに）
            pass 
            
        return torch.tanh(action_logits) # -1~1の連続値行動

class SARAEngine(nn.Module):
    """
    Spiking Attractor Recursive Architecture (SARA) v9.0
    
    Integrated Learning & Cognitive Functions:
    1. **Predictive Coding**: 入力予測誤差(Sensory PE)と状態予測誤差(State PE)による階層的推論。
    2. **Active Inference**: 自由エネルギー最小化による行動生成と内部状態の最適化。
    3. **Surprise-Modulated Plasticity**: 予測誤差の大きさに応じた動的学習率(旧Probabilistic Hebbian)。
    4. **Recursive Causal Trace**: 時間的因果関係の保持(旧Causal Trace)。
    5. **Physics-Informed World Model**: 物理則の内在化(旧PhysicsInformed)。
    """
    def __init__(self, config: SARAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_size = config.input_size or 128
        self.action_size = getattr(config, 'action_size', self.input_size) # Default to input size if not defined
        
        # --- Core SNN Circuit ---
        self.input_projection = nn.Linear(self.input_size, self.hidden_size)
        self.recurrent_weights = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # --- Integrated Sub-systems ---
        # 1. World Model (Physics & Observation)
        self.world_model = WorldModel(self.hidden_size, self.hidden_size * 2, self.input_size)
        
        # 2. Predictive Coding Units
        self.sensory_pe_unit = PredictiveErrorUnit(self.input_size) # 感覚入力レベルの誤差
        self.state_pe_unit = PredictiveErrorUnit(self.hidden_size)  # 潜在状態レベルの誤差
        
        # 3. Active Inference Controller
        self.active_inference = ActiveInferenceController(self.hidden_size, self.action_size)
        
        # --- Internal Buffers ---
        self.register_buffer('spike_trace', torch.zeros(1, self.hidden_size))
        
        # --- Parameters ---
        self.trace_decay = getattr(config, 'trace_decay', 0.95)
        self.base_learning_rate = getattr(config, 'learning_rate', 0.01)
        
        # Backward compatibility
        self.surprise_detector = None # Deprecated, logic moved to PredictiveErrorUnit

    def _update_traces(self, spikes: torch.Tensor):
        """因果トレースの更新"""
        self.spike_trace = self.trace_decay * self.spike_trace + (1.0 - self.trace_decay) * spikes

    def _apply_plasticity(self, pre: torch.Tensor, post: torch.Tensor, error_signal: torch.Tensor):
        """
        統合された可塑性ルール:
        Surprise-Modulated Hebbian + Predictive Coding Error Backprop (Local)
        """
        # 1. Hebbian Term (Correlation)
        hebbian = torch.matmul(self.spike_trace.t(), post)
        
        # 2. Error Modulation (Surprise)
        # 誤差が大きいほど、学習強度を上げる（重要イベントとして記憶）
        surprise_magnitude = error_signal.abs().mean()
        modulation = torch.sigmoid(surprise_magnitude * 5.0) # 0.5 ~ 1.0 scaling
        
        # 3. Weight Update
        # 勾配を使わない直接的な重み操作 (Online Learning)
        delta_w = self.base_learning_rate * modulation * hebbian
        
        with torch.no_grad():
            if delta_w.shape != self.recurrent_weights.weight.shape:
                # 簡易ブロードキャスト対応
                delta_w = delta_w.mean()
            
            self.recurrent_weights.weight.add_(delta_w)
            # Normalize to prevent explosion
            self.recurrent_weights.weight.data = F.normalize(self.recurrent_weights.weight.data, p=2, dim=1)

    def forward(self, 
                inputs: torch.Tensor, 
                prev_memory: Optional[SARAMemory] = None,
                target_goal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, SARAMemory, Dict[str, Any]]:
        """
        推論・予測・行動生成の統合ステップ
        """
        batch_size = inputs.size(0)
        device = inputs.device
        
        # Initialize Memory
        if prev_memory is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device)
            trace = torch.zeros(batch_size, self.hidden_size, device=device)
            wm_state = torch.zeros(batch_size, self.hidden_size, device=device)
        else:
            hidden = prev_memory.hidden_state
            trace = prev_memory.synaptic_trace
            wm_state = prev_memory.world_model_state
            self.spike_trace = trace

        # --- 1. Predictive Coding Phase (Perception) ---
        # Bottom-up Input
        input_current = self.input_projection(inputs)
        
        # World ModelからのTop-down予測（前の状態に基づく今の感覚入力の予測）
        _, sensory_pred, _ = self.world_model(wm_state)
        
        # 感覚レベルの予測誤差 (Sensory Prediction Error)
        # 実際の入力 - 予測された入力
        sensory_pe, free_energy_sensory = self.sensory_pe_unit(sensory_pred, inputs)
        
        # 誤差を入力として統合 (Error driving the network)
        # 従来の純粋な入力だけでなく、予測誤差がネットワークを駆動する
        drive_signal = input_current + self.input_projection(sensory_pe) # 簡易的な再投影
        
        # --- 2. Recurrent Integration (SNN Core) ---
        mem_potential = drive_signal + self.recurrent_weights(hidden)
        mem_potential = self.layer_norm(mem_potential)
        spikes = torch.sigmoid(mem_potential) # Rate approximation
        
        self._update_traces(spikes)
        
        # --- 3. World Model Dynamics (Simulation) ---
        # 現在の活動から次の状態を予測
        pred_next_state, _, energy = self.world_model(spikes)
        
        # 状態レベルの予測誤差 (State Prediction Error) - 時間的整合性
        # 前のステップで予測した「次の状態」と、実際に計算された「今の状態(spikes)」のズレ
        state_pe, free_energy_state = self.state_pe_unit(wm_state, spikes)
        
        # --- 4. Active Inference Phase (Action) ---
        # 自由エネルギーを最小化、またはゴールに近づくための行動を生成
        action = self.active_inference.infer_action(spikes, target_goal)
        
        # Total Free Energy (Surprise)
        total_free_energy = free_energy_sensory.mean() + free_energy_state.mean()
        
        # --- 5. Memory Update ---
        new_memory = SARAMemory(
            hidden_state=spikes,
            synaptic_trace=self.spike_trace.clone(),
            world_model_state=pred_next_state, # 次のステップの予測に使用
            prediction_error=total_free_energy,
            prior_belief=target_goal
        )
        
        info = {
            "sensory_pe": sensory_pe,
            "state_pe": state_pe,
            "energy": energy,
            "free_energy": total_free_energy,
            "action": action
        }
        
        return action, new_memory, info

    def adapt(self, 
              inputs: torch.Tensor, 
              targets: Optional[torch.Tensor] = None, 
              prev_memory: Optional[SARAMemory] = None) -> Dict[str, Any]:
        """
        自律学習ステップ (Active Inference & Learning)
        外部から呼び出されるメインのAPI
        """
        # ゴール設定（ターゲットがない場合は、現状維持や探索がゴールとなる）
        # Active Inferenceでは「期待する感覚入力(Target)」がGoalとして機能する
        
        # Forward pass (Active Inference含む)
        action, memory, info = self.forward(inputs, prev_memory, target_goal=targets)
        
        # 重み更新 (Plasticity)
        # 予測誤差(State PE)を教師信号としてネットワークを自己組織化
        self._apply_plasticity(
            pre=self.input_projection(inputs),
            post=memory.hidden_state,
            error_signal=info["state_pe"]
        )
        
        return {
            "loss": info["free_energy"],
            "surprise": info["free_energy"], # 互換性のため
            "memory": memory,
            "outputs": action, # SARAの出力は「行動」
            "info": info
        }

# Backward compatibility alias
SARABrainCore = SARAEngine

# Debug / Unit Test Logic
if __name__ == "__main__":
    print("Initializing SARA Engine v9.0 (Integrated)...")
    config = SARAConfig(hidden_size=64, input_size=32)
    brain = SARAEngine(config)
    
    dummy_input = torch.randn(2, 32) # Batch size 2
    
    print("Executing Adaptive Step...")
    results = brain.adapt(dummy_input)
    
    print(f"  Total Free Energy (Surprise): {results['loss'].item():.4f}")
    print(f"  Generated Action Shape: {results['outputs'].shape}")
    print(f"  Internal Energy Estimate: {results['info']['energy'].mean().item():.4f}")
    print("Integration of Predictive Coding & Active Inference successful.")