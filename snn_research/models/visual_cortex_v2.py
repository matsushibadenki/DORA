# ファイルパス: snn_research/models/visual_cortex_v2.py
# 日本語タイトル: visual_cortex_v2
# 目的: 線形誤差更新による学習の加速とマージン拡大 (Rev32)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, cast, List

from snn_research.core.snn_core import SpikingNeuralSubstrate
from snn_research.learning_rules.forward_forward import ForwardForwardRule
import logging

logger = logging.getLogger(__name__)

class VisualCortexV2(nn.Module):
    """
    Visual Cortex V2 - Phase 2 Rev32 (Linear Drive)
    
    修正内容:
    - 誤差計算を Sigmoid から Linear (ReLU) に変更。
      -> 閾値から離れていても学習が止まらないようにする（勾配消失の防止）。
    - 閾値を 150.0 -> 600.0 に引き上げ、現在の活動レベルに合わせる。
    - Vector Length を 30.0 -> 50.0 に拡大し、分離の余地を作る。
    """

    def __init__(self, device: torch.device, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.device = device
        self.config = config or {}

        self.input_dim = self.config.get("input_dim", 794)
        self.hidden_dim = self.config.get("hidden_dim", 2000)
        self.num_layers = self.config.get("num_layers", 3)
        
        self.config.setdefault("dt", 1.0)
        self.config.setdefault("tau_mem", 20.0)
        
        self.learning_rate = self.config.get("learning_rate", 0.05) # 少し下げる（Linearは値が大きくなりやすいため）
        
        # Vector Length: 50.0 -> Max Energy = 2500
        # 期待される平均活動 ~ 1250
        self.vector_length = 50.0
        
        # 閾値を中央付近に設定
        self.ff_threshold = self.config.get("ff_threshold", 600.0) 
        self.sparsity = 0.5 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]
        
        self._build_architecture()
        
        self.activity_history: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.layer_traces: Dict[str, torch.Tensor] = {}

    def _build_architecture(self):
        self.substrate.add_neuron_group("Retina", self.input_dim)

        prev_layer = "Retina"
        for i, layer_name in enumerate(self.layer_names):
            self.substrate.add_neuron_group(layer_name, self.hidden_dim)

            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold,
                w_decay=0.01 
            )

            projection_name = f"{prev_layer.lower()}_to_{layer_name.lower()}"
            self.substrate.add_projection(
                projection_name, prev_layer, layer_name, plasticity_rule=ff_rule
            )

            with torch.no_grad():
                proj = self.substrate.projections[projection_name]
                synapse = cast(nn.Module, proj.synapse)
                if hasattr(synapse, 'weight'):
                    w = cast(torch.Tensor, synapse.weight)
                    nn.init.orthogonal_(w, gain=1.0)
            
            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        
        # 入力正規化
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8) * self.vector_length
        
        inputs = {"Retina": x_norm}
        
        learning_phase = "neutral"
        inject_noise = False

        if phase == "wake":
            learning_phase = "positive"
            inject_noise = True
        elif phase == "sleep":
            learning_phase = "negative"
            inject_noise = True

        batch_size = x.size(0)
        if inject_noise:
            for name in self.layer_names:
                noise = torch.randn(batch_size, self.hidden_dim, device=self.device) * 0.5
                if name not in inputs:
                    inputs[name] = noise
                else:
                    inputs[name] += noise

        simulation_steps = 6
        last_out = {}
        self.layer_traces = {name: torch.zeros(batch_size, self.hidden_dim, device=self.device) 
                             for name in self.layer_names}

        for t in range(simulation_steps):
            current_phase = learning_phase if t >= 3 else "neutral"
            current_input = inputs["Retina"]
            
            for i, layer_name in enumerate(self.layer_names):
                prev_name = "Retina" if i == 0 else self.layer_names[i-1]
                
                proj_name = f"{prev_name.lower()}_to_{layer_name.lower()}"
                proj = self.substrate.projections[proj_name]
                synapse = cast(nn.Module, proj.synapse)
                
                weight = synapse.weight
                mem = F.linear(current_input, weight)
                
                # L2 Normalization
                norm = mem.norm(p=2, dim=1, keepdim=True) + 1e-8
                mem_normalized = mem / norm * self.vector_length
                
                activity = torch.relu(mem_normalized)
                self.layer_traces[layer_name] = 0.6 * self.layer_traces[layer_name] + 0.4 * activity
                
                current_input = activity.detach() 

                if t == simulation_steps - 1:
                    rate = (activity > 0).float().mean().item()
                    self.activity_history[layer_name] = 0.9 * self.activity_history[layer_name] + 0.1 * rate
        
        # Update
        if learning_phase != "neutral" and phase != "inference":
            with torch.no_grad():
                current_input = inputs["Retina"]
                
                for i, layer_name in enumerate(self.layer_names):
                    prev_name = "Retina" if i == 0 else self.layer_names[i-1]
                    proj_name = f"{prev_name.lower()}_to_{layer_name.lower()}"
                    proj = self.substrate.projections[proj_name]
                    synapse = cast(nn.Module, proj.synapse)
                    
                    v_activity = self.layer_traces[layer_name]
                    goodness = v_activity.pow(2).sum(dim=1, keepdim=True)
                    
                    # 【重要修正】Linear Error Calculation
                    # Sigmoidによる飽和を防ぎ、常に強い勾配を与える
                    if learning_phase == "positive":
                        # Posは閾値より大きくしたい。不足分(Threshold - Goodness)をエラーとする。
                        # 上限を設けないと不安定になるので、適度にクリップするか、ReLUを使う
                        # Goodness > Threshold の場合も、さらに大きくしてマージンを稼ぐ（過学習リスクはあるが分離優先）
                        raw_error = self.ff_threshold - goodness
                        
                        # 閾値以下なら強くプラス、閾値以上でも少しプラス（マージン確保）
                        # ここではシンプルに閾値との差分を使う
                        scale = raw_error # 正なら上げろ、負なら下げろ（いやPosは常に上げたい）
                        
                        # Hinton方式: log(1 + exp(...)) の微分は sigmoid。
                        # ここでは、Posをとにかく「上げる」方向に一定の力をかける
                        # ただし、Goodnessが極端に大きい場合は抑える
                        scale = 1.0 # 常に上げる
                        
                        # もっとスマートな方法:
                        # 閾値を超えていない分だけ強く押す
                        scale = torch.relu(self.ff_threshold + 200.0 - goodness) / 100.0
                        direction = 1.0
                    else:
                        # Negは閾値より小さくしたい。超過分(Goodness - Threshold)をエラーとする。
                        # 下限なしで押し下げる
                        scale = torch.relu(goodness - (self.ff_threshold - 200.0)) / 100.0
                        direction = -1.0
                    
                    mean_scale = scale.mean()
                    
                    if i == 0:
                        x_in = inputs["Retina"]
                    else:
                        x_in = self.layer_traces[prev_name]
                    
                    delta_w = (v_activity.t() @ x_in) / batch_size
                    
                    lr = self.learning_rate
                    # scaleが0になることもあるので、微小なベース学習率を持たせるかどうか
                    # ここではscaleに依存させる
                    
                    synapse.weight.add_(delta_w * direction * mean_scale * lr)
                    
                    w_norm = synapse.weight.norm(dim=1, keepdim=True) + 1e-8
                    synapse.weight.div_(w_norm).mul_(torch.clamp(w_norm, max=2.0))
                    
                    current_input = v_activity

        return last_out

    def get_goodness(self) -> Dict[str, float]:
        stats = {}
        for name in self.layer_names:
            if name in self.layer_traces:
                trace = self.layer_traces[name]
                goodness = trace.pow(2).sum(dim=1).mean().item()
            else:
                goodness = 0.0
            stats[f"{name}_goodness"] = goodness
        return stats

    def get_stability_metrics(self) -> Dict[str, float]:
        metrics = {}
        for name in self.layer_names:
            metrics[f"{name}_firing_rate"] = self.activity_history[name]
        return metrics

    def reset_state(self):
        self.layer_traces = {}