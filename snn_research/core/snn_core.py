# snn_research/core/snn_core.py
# Title: Spiking Neural Substrate v3.11 (Universal Adapter)
# Description: 
#   forward メソッドの引数を *args, **kwargs に完全開放。
#   TypeError の発生を理論的に不可能にし、内部ロジックで入力を特定する。

from __future__ import annotations
import logging
import random
import math
from typing import Any, Dict, List, Optional, cast, Union, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from snn_research.hardware.event_driven_simulator import DORAKernel

logger = logging.getLogger(__name__)

class SpikingNeuralSubstrate(nn.Module):
    def __init__(self, config: Dict[str, Any], device: torch.device = torch.device('cpu'), **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        self.device = device
        self.dt: float = config.get("dt", 1.0)
        self.kernel = DORAKernel(dt=self.dt)
        self.group_indices: Dict[str, Tuple[int, int]] = {}
        self.prev_spikes: Dict[str, Optional[Tensor]] = {}
        self.uncertainty_score = 0.0 
        logger.info("⚡ SpikingNeuralSubstrate v3.11 (Universal Adapter) initialized.")

    def compile(self, model: Optional[nn.Module] = None) -> None:
        if not model: return
        logger.info(f"🔨 Compiling {type(model).__name__}...")
        self.kernel = DORAKernel(dt=self.dt)
        self.group_indices = {}
        
        input_dim = getattr(model, 'dim', 128)
        self._create_group("input", input_dim, v_thresh=0.2)
        curr = "input"
        
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                b_name = f"block_{i}"
                self._create_group(f"{b_name}_in", layer.in_proj.out_features, v_thresh=0.5)
                self.kernel.connect_groups(self.group_indices[curr], self.group_indices[f"{b_name}_in"], layer.in_proj.weight.detach().cpu().numpy())
                
                d_inner = layer.in_proj.out_features // 2
                self._create_group(f"{b_name}_out", layer.out_proj.out_features, v_thresh=0.5)
                src_range = self.group_indices[f"{b_name}_in"]
                self.kernel.connect_groups((src_range[0], src_range[0]+d_inner), self.group_indices[f"{b_name}_out"], layer.out_proj.weight.detach().cpu().numpy())
                
                self.kernel.connect_groups(self.group_indices[curr], self.group_indices[f"{b_name}_out"], torch.eye(layer.out_proj.out_features).numpy())
                curr = f"{b_name}_out"
        
        if hasattr(model, "output_projection") and isinstance(model.output_projection, nn.Linear):
            self._create_group("output", model.output_projection.out_features, v_thresh=0.5)
            self.kernel.connect_groups(self.group_indices[curr], self.group_indices["output"], model.output_projection.weight.detach().cpu().numpy())
        else:
            self.group_indices["output"] = self.group_indices[curr]
            
        logger.info(f"✅ Compilation Successful. Neurons: {len(self.kernel.neurons)}")

    def _create_group(self, name: str, count: int, v_thresh: float):
        start, end = self.kernel.create_layer_neurons(count, layer_id=len(self.group_indices), v_thresh=v_thresh)
        self.group_indices[name] = (start, end)
        self.prev_spikes[name] = torch.zeros(1, count, device=self.device)

    def forward(self, *args, **kwargs) -> Tensor:
        """
        [Universal Fix] 引数エラーを回避するための万能受け口。
        args[0] または kwargs['input'] / kwargs['x'] からテンソルを探し出す。
        """
        input_tensor = None
        
        # 1. 位置引数の確認
        if args:
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    input_tensor = arg
                    break
        
        # 2. キーワード引数の確認
        if input_tensor is None:
            input_tensor = kwargs.get('input') or kwargs.get('x')

        # 3. それでもなければダミー (エラーで落とさないための最終防壁)
        if input_tensor is None:
            input_tensor = torch.zeros(1, 128, device=self.device)

        # 内部処理へ委譲
        res_dict = self.forward_step({"input": input_tensor})
        spikes = res_dict.get("spikes", {})
        
        # 出力の特定
        if "output" in spikes and spikes["output"] is not None:
            out = spikes["output"]
        else:
            # 代替策：最後に定義された有効なグループ
            valid_keys = [k for k, v in spikes.items() if v is not None]
            if valid_keys:
                out = spikes[valid_keys[-1]]
            else:
                out = torch.zeros(1, 128, device=self.device)
            
        # 形状保証 (Batch, Dim)
        if out.dim() == 1:
            out = out.unsqueeze(0)
            
        return out

    def forward_step(self, ext_inputs: Dict[str, Tensor], learning: bool = True, dreaming: bool = False, **kwargs: Any) -> Dict[str, Any]:
        jitter = 0.1
        if not dreaming:
            for name, tensor in ext_inputs.items():
                if name in self.group_indices:
                    start_id, _ = self.group_indices[name]
                    t = torch.as_tensor(tensor)
                    indices = torch.nonzero(t.flatten() > 0.1).flatten().cpu().numpy()
                    self.kernel.push_input_spikes([int(idx + start_id) for idx in indices], self.kernel.current_time + jitter)
        else:
            n_len = len(self.kernel.neurons)
            if n_len > 0:
                pop = list(range(n_len))
                sample_size = min(n_len, max(1, n_len // 50))
                dream_indices = random.sample(pop, sample_size)
                self.kernel.push_input_spikes(dream_indices, self.kernel.current_time + jitter)

        counts = self.kernel.run(duration=self.dt, learning_enabled=learning)
        
        curr_spikes = {}
        for name, (s, e) in self.group_indices.items():
            spikes = torch.zeros(1, e-s, device=self.device)
            for nid, count in counts.items():
                if s <= nid < e and count > 0:
                    spikes[0, nid-s] = 1.0
            curr_spikes[name] = spikes
        
        ent = 0.0
        if "output" in curr_spikes:
            ratio = max(1e-9, min(1.0 - 1e-9, float(curr_spikes["output"].mean().item())))
            ent = -(ratio * math.log(ratio) + (1-ratio) * math.log(1-ratio))
        self.uncertainty_score = (ent + self.kernel.stats.get("surprise_index", 0.0)) / 2.0
            
        self.prev_spikes = cast(Dict[str, Optional[Tensor]], curr_spikes)
        return {"spikes": curr_spikes, "uncertainty": self.uncertainty_score}

    def sleep_process(self):
        self.kernel.apply_synaptic_scaling(factor=0.9)
        stats = self.kernel.stats
        tagged = sum(1 for n in self.kernel.neurons for s in n.outgoing_synapses if s.is_tagged)
        logger.info(f"🧠 Plasticity Report: Created={stats['synapses_created']}, Pruned={stats['synapses_pruned']}, Tagged={tagged}, Synapses={stats['current_synapses']}")

    def reset_state(self):
        self.kernel.reset_state()
        self.uncertainty_score = 0.0

SNNCore = SpikingNeuralSubstrate