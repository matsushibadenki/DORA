# directory: snn_research/core
# filename: factories.py
# description: 修正版 (インポートエラー対策追加)

from typing import Dict, Any, Optional
import torch.nn as nn

# 他のモデル
from snn_research.models.transformer.spikformer import Spikformer
from snn_research.models.adapters.sara_adapter import SaraAdapter

# TemporalSNNのインポート (エラーハンドリング付き)
try:
    from snn_research.models.bio.temporal_snn import TemporalSNN
except ImportError as e:
    print(f"[Warning] TemporalSNN import failed in factories: {e}")
    TemporalSNN = None

def create_model(config: Dict[str, Any]) -> nn.Module:
    model_type = config.get("model_type", "snn").lower()
    
    # SARA Engine
    if model_type in ["sara", "sara_engine", "online_rl_snn", "attractor_memory"]:
        return SaraAdapter(config)
        
    # Spikformer
    if model_type == "spikformer":
        return Spikformer(**config.get("model_params", {}))
    
    # Temporal SNN
    elif model_type == "temporal_snn":
        if TemporalSNN is not None:
            return TemporalSNN(config.get("model_params", {}))
        else:
            print("[Error] TemporalSNN requested but module not available. Falling back to SARA.")
            return SaraAdapter(config)
            
    # Default
    from snn_research.models.bio.simple_network import SimpleSNN
    return SimpleSNN(config)