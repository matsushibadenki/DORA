# ファイルパス: snn_research/models/visual_cortex_v2.py
# 日本語タイトル: visual_cortex_v2
# 目的: V1層のLayerNorm廃止による画像信号の復権（Rev29）

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
    Visual Cortex V2 - Phase 2 Rev29 (Sensory Dominance)
    
    修正内容:
    - V1層の LayerNorm を廃止。
      -> 画像入力のエネルギーを直接ニューロン活動に反映させ、
         「画像が変われば活動が変わる」ことを物理的に強制する。
    - V1へのラベル注入を極小化 (Gain 0.05)。
      -> V1はほぼ純粋な視覚野として振る舞い、Pos/Negの区別はわずかなバイアスで行う。
    - V2/V3は LayerNorm を維持し、抽象化と統合を行う。
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
        
        self.learning_rate = self.config.get("learning_rate", 0.05)
        # V1は正規化なしで大きな値になる可能性があるため、閾値を層ごとに管理するのが理想だが
        # ここでは全体的に少し高めに設定しておく
        self.ff_threshold = self.config.get("ff_threshold", 1500.0) 
        self.sparsity = 0.5 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]
        
        self.label_projections = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()

        for name in self.layer_names:
            self.label_projections[name] = nn.Linear(10, self.hidden_dim, bias=False)
            
            # 【重要修正】V1以外のみLayerNormを適用
            if name != "V1":
                self.layer_norms[name] = nn.LayerNorm(self.hidden_dim, elementwise_affine=True)
        
        self._build_architecture()
        
        self.activity_history: Dict[str, float] = {name: 0.0 for name in self.layer_names}
        self.layer_traces: Dict[str, torch.Tensor] = {}

    def _build_architecture(self):
        self.substrate.add_neuron_group("Retina", 784)

        prev_layer = "Retina"
        for i, layer_name in enumerate(self.layer_names):
            self.substrate.add_neuron_group(layer_name, self.hidden_dim)

            # V1の重みは減衰させない（入力を維持するため）
            decay = 0.0 if layer_name == "V1" else 0.02
            
            ff_rule = ForwardForwardRule(
                learning_rate=self.learning_rate,
                threshold=self.ff_threshold,
                w_decay=decay
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
                    # 初期結合強度
                    nn.init.orthogonal_(w, gain=1.2)
            
            prev_layer = layer_name
            
        with torch.no_grad():
            for name, proj in self.label_projections.items():
                if name == "V1":
                    # 【重要修正】V1へのラベル影響は極小にする (0.05)
                    # これにより、V1の活動の95%以上は画像由来となる
                    nn.init.orthogonal_(proj.weight, gain=0.05)
                else:
                    # 後続層はしっかり統合する
                    nn.init.orthogonal_(proj.weight, gain=0.5)

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        
        img = x[:, :784]
        lbl = x[:, 784:]
        
        # 入力ゲイン: V1にLayerNormがないため、この値が直接Goodnessの大きさに直結する
        # sqrt(784) * 1.5 ~= 40. Energy ~= 1600. Threshold 1500とマッチする。
        img = img / (img.norm(p=2, dim=1, keepdim=True) + 1e-8) * 1.5
        
        batch_size = x.size(0)
        label_currents = {}
        for name in self.layer_names:
            label_currents[name] = self.label_projections[name](lbl)

        inputs = {"Retina": img}
        
        learning_phase = "neutral"
        inject_noise = False

        if phase == "wake":
            learning_phase = "positive"
            inject_noise = True
        elif phase == "sleep":
            learning_phase = "negative"
            inject_noise = True

        if inject_noise:
            for name in self.layer_names:
                # ノイズもV1は控えめに
                scale = 0.05 if name == "V1" else 0.1
                noise = torch.randn(batch_size, self.hidden_dim, device=self.device) * scale
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
            out = self.substrate.forward_step(inputs, phase=current_phase)
            last_out = out

            with torch.no_grad():
                for name in self.layer_names:
                    group = self.substrate.neuron_groups[name]
                    
                    if hasattr(group, "mem"):
                        mem = cast(torch.Tensor, group.mem)
                        mem.add_(label_currents[name])
                        
                        # V1は正規化しない（Raw Signal）
                        if name == "V1":
                            activity = torch.relu(mem)
                        else:
                            # V2/V3は正規化して安定化
                            normalized_mem = self.layer_norms[name](mem)
                            activity = torch.relu(normalized_mem)
                            
                        self.layer_traces[name] = 0.6 * self.layer_traces[name] + 0.4 * activity

                        if t == simulation_steps - 1:
                            rate = (activity > 0).float().mean().item()
                            self.activity_history[name] = 0.9 * self.activity_history[name] + 0.1 * rate
        
        # Update Label Weights
        if learning_phase != "neutral" and phase != "inference":
            with torch.no_grad():
                lr_label = 0.05
                
                for name in self.layer_names:
                    v_activity = self.layer_traces[name]
                    goodness = v_activity.pow(2).sum(dim=1, keepdim=True)
                    
                    if learning_phase == "positive":
                        error = torch.sigmoid(self.ff_threshold - goodness)
                        direction = 1.0
                    else:
                        error = torch.sigmoid(goodness - self.ff_threshold)
                        direction = -1.0
                    
                    mean_error = error.mean()
                    
                    # (Hidden, Batch) @ (Batch, 10) -> (Hidden, 10)
                    delta_w = (v_activity.t() @ lbl) / batch_size
                    
                    proj = self.label_projections[name]
                    proj.weight.add_(delta_w * direction * mean_error * lr_label)
                    
                    norm = proj.weight.norm(dim=1, keepdim=True) + 1e-8
                    # V1のラベル重みは小さく保つ
                    limit = 0.5 if name == "V1" else 3.0
                    proj.weight.div_(norm).mul_(torch.clamp(norm, max=limit))

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
        self.substrate.reset_state()
        self.layer_traces = {}