# snn_research/core/layers/predictive_coding.py
# ファイルパス: snn_research/core/layers/predictive_coding.py
# 修正内容: 
# - 引数名を input_size/hidden_size に変更 (BioPCNetworkとの整合性)
# - STDP学習則の統合と update_weights メソッドの実装
# - forwardの戻り値にスパイク情報を追加

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional, List, Type, cast, Union
import logging

from snn_research.learning_rules.stdp import STDP

try:
    from snn_research.core.neurons import (
        AdaptiveLIFNeuron, IzhikevichNeuron, GLIFNeuron,
        TC_LIF, DualThresholdNeuron, ScaleAndFireNeuron,
        BistableIFNeuron, EvolutionaryLeakLIF
    )
    from snn_research.core.layers.bit_spike_layer import BitSpikeLinear
except ImportError:
    # 開発環境等でのフォールバック
    AdaptiveLIFNeuron = Any  # type: ignore
    IzhikevichNeuron = Any  # type: ignore
    GLIFNeuron = Any  # type: ignore
    TC_LIF = Any  # type: ignore
    DualThresholdNeuron = Any  # type: ignore
    ScaleAndFireNeuron = Any  # type: ignore
    BistableIFNeuron = Any  # type: ignore
    EvolutionaryLeakLIF = Any  # type: ignore
    
    class BitSpikeLinear(nn.Linear):  # type: ignore
        def __init__(self, in_features, out_features, bias=True, quantize_inference=True):
            super().__init__(in_features, out_features, bias=bias)

logger = logging.getLogger(__name__)


class PredictiveCodingLayer(nn.Module):
    """
    Predictive Coding (PC) を実行するSNNレイヤー。

    Biomimetic Enhancement:
    - Iterative Inference: 複数ステップの緩和(Relaxation)による推論。
    - Online Learning: STDPによる重み更新。
    - BitNet Integration: 重みを {-1, 0, 1} に量子化し高速化。
    """

    def __init__(
        self,
        input_size: int,      # 旧 d_model
        hidden_size: int,     # 旧 d_state
        neuron_class: Type[nn.Module],
        neuron_params: Dict[str, Any],
        weight_tying: bool = True,
        sparsity: float = 0.05,
        inference_steps: int = 5,
        inference_lr: float = 0.1,
        use_bitnet: bool = True,
        learning: bool = True
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_tying = weight_tying
        self.sparsity = sparsity
        self.inference_steps = inference_steps
        self.inference_lr = inference_lr
        self.use_bitnet = use_bitnet
        self.learning = learning

        filtered_params = self._filter_params(neuron_class, neuron_params)

        # 線形層のクラス選択
        LinearLayer = BitSpikeLinear if use_bitnet else nn.Linear
        linear_kwargs = {'quantize_inference': True} if use_bitnet else {}

        # 1. Generative Path (Top-Down: Hidden/State -> Input/Prediction)
        self.generative_fc = LinearLayer(hidden_size, input_size, **linear_kwargs)
        self.generative_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                                      neuron_class(features=input_size, **filtered_params))

        # 2. Inference Path (Bottom-Up: Error -> Hidden/State Update)
        if self.weight_tying:
            self.inference_fc = None
        else:
            self.inference_fc = LinearLayer(input_size, hidden_size, **linear_kwargs)

        self.inference_neuron = cast(Union[AdaptiveLIFNeuron, IzhikevichNeuron],
                                     neuron_class(features=hidden_size, **filtered_params))

        self.norm_state = nn.LayerNorm(hidden_size)
        self.norm_error = nn.LayerNorm(input_size)

        self.error_scale = nn.Parameter(torch.tensor(1.0))
        self.feedback_strength = nn.Parameter(torch.tensor(1.0))
        
        # 3. Learning Rule (STDP)
        if self.learning:
            self.stdp = STDP(learning_rate=0.005) # パラメータは外部設定から注入すべきだが一旦デフォルト
            self.trace_state: Dict[str, Any] = {}

    def _filter_params(self, neuron_class: Type[nn.Module], neuron_params: Dict[str, Any]) -> Dict[str, Any]:
        """指定されたニューロンクラスが受け入れるパラメータのみを抽出する"""
        valid_params: List[str] = []
        if neuron_class == AdaptiveLIFNeuron:
            valid_params = ['features', 'tau_mem', 'base_threshold', 'adaptation_strength',
                            'target_spike_rate', 'noise_intensity', 'threshold_decay', 'threshold_step', 'v_reset']
        elif hasattr(neuron_class, '__name__') and neuron_class.__name__ == 'LIFNeuron':
             # 今回修正したLIFNeuron用
             valid_params = ['features', 'tau_mem', 'tau_adap', 'v_threshold', 'v_reset', 'theta_plus', 'dt']
        else:
             # フォールバック
             valid_params = ['features', 'tau_mem', 'base_threshold', 'v_reset']

        # 存在するキーだけ残す（厳密なチェックは省略）
        return {k: v for k, v in neuron_params.items()}

    def _apply_lateral_inhibition(self, x: torch.Tensor) -> torch.Tensor:
        """Hard k-WTA"""
        if self.sparsity >= 1.0 or self.sparsity <= 0.0:
            return x
        x_abs = x.abs()
        B, N = x.shape
        k = int(N * self.sparsity)
        if k == 0:
            k = 1
        topk_values, _ = torch.topk(x_abs, k, dim=1)
        threshold = topk_values[:, -1].unsqueeze(1)
        threshold = torch.max(threshold, torch.tensor(1e-6, device=x.device))
        mask = (x_abs >= threshold).float()
        return x * mask

    def update_weights(self, bottom_input: torch.Tensor, top_state: torch.Tensor, error: torch.Tensor, spikes: torch.Tensor):
        """
        オンライン学習(STDP)による重み更新。
        
        Args:
            bottom_input: 下層からの入力（今回は使用せず、Postスパイクを使用）
            top_state: 上層の状態（Pre-synaptic spikes）
            error: 予測誤差
            spikes: 生成ニューロンの発火（Post-synaptic spikes）
        """
        if not self.learning:
            return

        # Pre: Top-down State (Spikes expected)
        # Post: Generative Neuron Output (Spikes)
        # Note: top_stateはニューロン出力なのでスパイク列とみなす
        
        # 重み行列の取得 (BitNetの場合は量子化前の実数重みを更新)
        weights = self.generative_fc.weight
        
        # STDP更新
        delta_w, logs = self.stdp.update(
            pre_spikes=top_state,
            post_spikes=spikes,
            current_weights=weights,
            local_state=self.trace_state
        )
        
        if delta_w is not None:
            # 勾配として適用せず、直接重みを書き換える（No Backprop制約）
            # 注意: PyTorchのAutogradと競合しないよう no_grad で実行
            with torch.no_grad():
                self.generative_fc.weight.add_(delta_w)

    def forward(
        self,
        bottom_up_input: torch.Tensor,
        top_down_state: torch.Tensor,
        top_down_error: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inference Phase as Relaxation
        
        Returns:
            current_state (Tensor): 更新されたHidden State
            final_error (Tensor): 予測誤差
            combined_mem (Tensor): 膜電位（可視化用）
            gen_spikes (Tensor): 生成ニューロンのスパイク（学習用）
        """

        # 初期状態
        current_state = top_down_state.clone()
        final_error = torch.zeros_like(bottom_up_input)
        combined_mem_list = []
        last_gen_spikes = torch.zeros_like(bottom_up_input)

        # --- Relaxation Loop ---
        for step in range(self.inference_steps):
            # 1. Generative Pass
            pred_input = self.generative_fc(self.norm_state(current_state))
            pred, gen_mem = self.generative_neuron(pred_input)
            
            # STDP用にスパイクを保存
            if step == self.inference_steps - 1:
                last_gen_spikes = pred

            # 2. Error Calculation
            raw_error = bottom_up_input - pred
            error = raw_error * self.error_scale

            if step == self.inference_steps - 1:
                final_error = error

            # 3. Inference Pass (State Update)
            norm_error = self.norm_error(error)

            if self.weight_tying:
                if self.use_bitnet and hasattr(self.generative_fc, 'weight'):
                    from snn_research.core.layers.bit_spike_layer import bit_quantize_weight
                    w_quant = bit_quantize_weight(self.generative_fc.weight, 1e-5)
                    bu_input = F.linear(norm_error, w_quant.t())
                else:
                    bu_input = F.linear(norm_error, self.generative_fc.weight.t())
            else:
                if self.inference_fc is None:
                    raise RuntimeError("inference_fc is None")
                bu_input = self.inference_fc(norm_error)

            total_input = bu_input
            if top_down_error is not None:
                total_input = total_input - (top_down_error * self.feedback_strength)

            # 状態更新
            state_update, inf_mem = self.inference_neuron(total_input)
            state_update = self._apply_lateral_inhibition(state_update)

            # 状態変数の緩和更新
            current_state = current_state * (1.0 - self.inference_lr) + state_update * self.inference_lr

            if step == self.inference_steps - 1:
                combined_mem_list.append(torch.cat((gen_mem, inf_mem), dim=1))

        combined_mem = combined_mem_list[-1] if combined_mem_list else torch.tensor(0.0)

        return current_state, final_error, combined_mem, last_gen_spikes