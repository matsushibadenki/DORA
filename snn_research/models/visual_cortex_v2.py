# ファイルパス: snn_research/models/visual_cortex_v2.py
# 日本語タイトル: visual_cortex_v2
# 目的: LayerNorm撤廃とk-WTAによるスパース分散表現の獲得 (Rev36)

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
    Visual Cortex V2 - Phase 2 Rev36 (Sparse Competitive Coding)
    
    修正内容:
    - LayerNormを撤廃し、k-WTA (k-Winners-Take-All) を復活。
      -> 正規化ではなく「競合」によって活動レベルを制御する。
      -> ラベル入力が「誰が勝つか」に直接影響を与えるため、特徴分離が強力になる。
    - 入力は画像とラベルを結合(Concat)して使用。
    - 学習則は閾値ベースの線形更新を維持。
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
        
        # k-WTA設定
        # 2000ニューロンのうち上位5%(100個)が発火すると想定
        # 活動値の平均が1.0なら Goodness = 1.0^2 * 100 = 100
        # 活動値が強い(例: 5.0)なら Goodness = 25 * 100 = 2500
        # 閾値を 800.0 に設定 (平均活動強度 ~2.8 を要求)
        self.sparsity = 0.05 
        self.ff_threshold = self.config.get("ff_threshold", 800.0) 

        self.substrate = SpikingNeuralSubstrate(self.config, self.device)
        self.layer_names = [f"V{i+1}" for i in range(self.num_layers)]
        
        # ラベル専用層は不要（入力統合のため）
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
                w_decay=0.02 # k-WTA環境下では重みが大きくなりがちなので減衰を少し強める
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
                    # k-WTAでは初期発火が必要なので、少し強めに初期化
                    nn.init.orthogonal_(w, gain=1.5)
            
            prev_layer = layer_name

    def forward(self, x: torch.Tensor, phase: str = "wake") -> Dict[str, torch.Tensor]:
        x = x.to(self.device).float()
        
        # 入力正規化: ベクトル長を固定するのではなく、適切なスケールに収める
        # 画像平均ノルム~9.0 -> 15.0程度にスケールアップ
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8) * 15.0
        
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
                # k-WTAのデッドロック防止のため、ノイズは重要
                noise = torch.randn(batch_size, self.hidden_dim, device=self.device) * 0.2
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
                
                # 【重要】k-WTA (k-Winners-Take-All)
                # 上位k個のニューロンのみを通す
                # これにより、活動の総量が制限され（爆発防止）、
                # かつニューロン間の競合により専門化（特徴抽出）が促進される
                k = int(self.hidden_dim * self.sparsity)
                if k > 0:
                    # topkを取得
                    topk_values, _ = torch.topk(mem, k, dim=1)
                    # k番目の値を閾値とする
                    threshold = topk_values[:, -1].unsqueeze(1)
                    # 閾値以上のものだけマスクを作成（バイナリマスクではない、値は保持）
                    # ReLUも兼ねて、正の値のみを通す（閾値が負の場合もあるため）
                    mask = (mem >= threshold).float()
                    activity = torch.relu(mem * mask)
                else:
                    activity = torch.relu(mem)
                
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
                    
                    # 線形更新則 (Rev32を踏襲)
                    # 閾値 800 に対して、Posは上げ、Negは下げる
                    if learning_phase == "positive":
                        scale = torch.sigmoid(self.ff_threshold - goodness)
                        direction = 1.0
                    else:
                        scale = torch.sigmoid(goodness - self.ff_threshold)
                        direction = -1.0
                    
                    mean_scale = scale.mean()
                    
                    if i == 0:
                        x_in = inputs["Retina"]
                    else:
                        x_in = self.layer_traces[prev_name]
                    
                    delta_w = (v_activity.t() @ x_in) / batch_size
                    
                    lr = self.learning_rate
                    synapse.weight.add_(delta_w * direction * mean_scale * lr)
                    
                    # k-WTAでは少数のニューロンに重みが集中しやすいため、
                    # 個別の重みベクトルの長さを正規化するのが有効
                    # 行ごと（ニューロンごと）に正規化
                    # w_norm = synapse.weight.norm(dim=1, keepdim=True) + 1e-8
                    # synapse.weight.div_(w_norm).mul_(torch.clamp(w_norm, max=3.0))
                    # 全体のクリッピングに変更
                    torch.nn.utils.clip_grad_norm_(synapse.parameters(), 1.0)
                    
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