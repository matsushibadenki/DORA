# ファイルパス: snn_research/models/bio/visual_cortex.py
# 日本語タイトル: Bio-Inspired Visual Cortex Model (Dynamic Shape Support)
# 目的・内容:
#   霊長類の視覚野（V1, V2, V4, IT）を模した階層型SNNモデル。
#   入力次元やチャネル数を動的に設定可能にし、時系列入力(Video)と静止画入力(Static)の両方に対応。

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union

from snn_research.core.base import BaseModel
from snn_research.core.networks.sequential_snn_network import SequentialSNN
from snn_research.core.layers.lif_layer import LIFLayer
import logging

logger = logging.getLogger(__name__)

class VisualCortex(BaseModel):
    """
    生物学的視覚野モデル。
    Retina -> V1 -> V2 -> V4 -> IT の階層処理を行う。
    """

    def __init__(
        self,
        input_shape: Tuple[int, int] = (28, 28), 
        in_channels: int = 1,
        base_channels: int = 64, 
        time_steps: int = 10,
        neuron_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        
        self.time_steps = time_steps
        self.input_shape = input_shape
        self.in_channels = in_channels
        
        # 入力次元の計算 (H * W * C)
        flat_input_dim = input_shape[0] * input_shape[1] * in_channels
        
        # 各層のニューロン数設定
        v1_dim = base_channels * 2
        v2_dim = base_channels * 4
        v4_dim = base_channels * 6
        it_dim = base_channels * 8 

        # ニューロン設定
        params = neuron_params or {}
        lif_config = {
            "decay": 0.9,
            "threshold": params.get("base_threshold", 1.0),
            "v_reset": 0.0,
            "tau_mem": params.get("tau_mem", 20.0)
        }
        lif_config.update(kwargs.get("lif_config", {}))

        # 階層の構築
        self.pathway = SequentialSNN([
            # V1: エッジ検出・基本特徴
            LIFLayer(input_features=flat_input_dim, neurons=v1_dim, name="V1", **lif_config),
            
            # V2: テクスチャ・複雑な形状
            LIFLayer(input_features=v1_dim, neurons=v2_dim, name="V2", **lif_config),
            
            # V4: 物体部分・色
            LIFLayer(input_features=v2_dim, neurons=v4_dim, name="V4", **lif_config),
            
            # IT: 物体全体・概念
            LIFLayer(input_features=v4_dim, neurons=it_dim, name="IT", **lif_config)
        ])

        logger.info(f"👁️ VisualCortex initialized: Input({flat_input_dim}) -> V1({v1_dim}) -> IT({it_dim})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        視覚処理の実行。
        Args:
            x: 
              - Static Image: [Batch, Channels, Height, Width]
              - Video: [Batch, Time, Channels, Height, Width]
        Returns:
            torch.Tensor: [Batch, Time, Features] (IT層の活動)
        """
        batch_size = x.shape[0]
        
        # 入力の形状確認と前処理
        if x.dim() == 5:
            # Video: [Batch, Time, C, H, W]
            time_steps = x.shape[1]
            # 各タイムステップごとにフラット化: [Batch, Time, Features]
            x_flat = x.view(batch_size, time_steps, -1)
            is_video = True
        elif x.dim() == 4:
            # Static Image: [Batch, C, H, W]
            time_steps = self.time_steps
            # フラット化して入力を用意: [Batch, Features]
            input_flat = x.view(batch_size, -1)
            x_flat = input_flat
            is_video = False
        else:
            # 既にフラットなどの場合
            if x.dim() == 2:
                time_steps = self.time_steps
                x_flat = x
                is_video = False
            elif x.dim() == 3:
                time_steps = x.shape[1]
                x_flat = x
                is_video = True
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}")

        outputs = []
        
        # 時間方向のループ処理
        for t in range(time_steps):
            # 現在のタイムステップの入力を取得
            if is_video:
                current_input = x_flat[:, t, :]
            else:
                current_input = x_flat # Staticの場合は同じ入力を継続注入

            # 順伝播
            step_output = self.pathway(current_input)
            outputs.append(step_output)

        # 時間方向にスタック: [Batch, Time, Features]
        output_stack = torch.stack(outputs, dim=1)
        
        return output_stack

    def reset_state(self) -> None:
        self.pathway.reset_state()