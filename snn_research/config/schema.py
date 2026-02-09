# directory: snn_research/config
# file: schema.py
# purpose: Configuration schemas for SNN models and SARA engine
# description: ニューロン設定やモデル設定、SARAエンジンの設定定義を行うデータクラス群

from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from omegaconf import MISSING

@dataclass
class NeuronConfig:
    type: str = "lif"
    tau_m_init: float = 2.0
    tau_s_init: float = 2.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    surrogate_function: str = "sigmoid"
    detach_reset: bool = True

@dataclass
class SARAConfig:
    """Configuration for SARA Engine (Spiking Attractor Recursive Architecture)"""
    hidden_size: int = 128
    input_size: Optional[int] = 128
    plasticity_mode: str = "surprise_modulated"  # 'stdp', 'hebbian', 'surprise_modulated'
    reasoning_depth: int = 3
    use_world_model: bool = True
    learning_rate: float = 0.01
    trace_decay: float = 0.95
    # World Model specific config
    world_model_hidden_dim: int = 256
    surprise_scale: float = 0.1

@dataclass
class ModelConfig:
    """General Model Configuration"""
    model_name: str = "snn_basic"
    hidden_dim: int = 128
    layer_count: int = 3
    neuron: NeuronConfig = field(default_factory=NeuronConfig)
    
@dataclass
class TrainingConfig:
    """Training Hyperparameters"""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    device: str = "cuda"  # or "cpu", "mps"