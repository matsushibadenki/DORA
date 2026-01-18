# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: Spiking Neural Substrate (The Kernel) v3.3
# 目的・内容:
#   Neuromorphic OSの中核となる神経基盤。
#   ニューロン集団とシナプス結合、および可塑性ルールを物理シミュレーションとして実行する。
#   ドキュメントの「Universal Neuron Kernel」に相当。

import logging
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple

# Import Native LIFNeuron
try:
    from snn_research.core.neurons.lif_neuron import LIFNeuron
except ImportError:
    # パス解決のフォールバック
    import sys
    import os
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")))
    from snn_research.core.neurons.lif_neuron import LIFNeuron

from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)


class SynapticProjection(nn.Module):
    """
    ニューロン集団間のシナプス結合を表現するクラス。
    学習則（PlasticityRule）を保持し、ローカルな更新を適用する。
    """

    def __init__(self, in_features: int, out_features: int, plasticity_rule: Optional[PlasticityRule] = None):
        super().__init__()
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.plasticity_rule = plasticity_rule

        # 直交初期化による信号伝播の安定化
        nn.init.orthogonal_(self.synapse.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """入力スパイク列に対するシナプス電流を計算"""
        return self.synapse(x)

    def apply_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor, **kwargs) -> Dict[str, Any]:
        """学習則の適用"""
        if self.plasticity_rule is None:
            return {}

        with torch.no_grad():
            # plasticity_rule.update は delta_w とログを返す
            delta_w, logs = self.plasticity_rule.update(
                pre_spikes=pre_spikes,
                post_spikes=post_spikes,
                current_weights=self.synapse.weight.data,
                **kwargs
            )

            if delta_w is not None:
                self.synapse.weight.data += delta_w
                # 重みの発散を防ぐためのクリッピング
                self.synapse.weight.data.clamp_(-5.0, 5.0)

        return logs


class SpikingNeuralSubstrate(nn.Module):
    """
    Neuromorphic OSのための汎用神経基盤（Kernel）。
    明示的な時間ステップまたはイベント駆動で状態を更新する。
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        super().__init__()
        self.config = config
        self.device = device

        self.time_step = 0
        self.dt = config.get("dt", 1.0)

        self.neuron_groups: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        self.topology: List[Dict[str, str]] = []

        # 前回のスパイク状態（STDP等で使用）
        self.prev_spikes: Dict[str, torch.Tensor] = {}

        logger.info("⚡ SpikingNeuralSubstrate initialized.")

    def add_neuron_group(self, name: str, num_neurons: int, neuron_model: Optional[nn.Module] = None):
        """
        ニューロン集団（領野）を追加する。
        """
        if neuron_model is None:
            neuron_model = LIFNeuron(
                features=num_neurons,
                tau_mem=self.config.get("tau_mem", 20.0),
                v_threshold=self.config.get("threshold", 1.0),
                dt=self.dt
            )

        # ニューロンが状態（膜電位）を維持するように設定
        if hasattr(neuron_model, "set_stateful"):
            neuron_model.set_stateful(True)  # type: ignore

        self.neuron_groups[name] = neuron_model.to(self.device)
        self.prev_spikes[name] = None

        logger.debug(f"  + Group added: {name} ({num_neurons} neurons)")

    def add_projection(self, name: str, source: str, target: str, plasticity_rule: Optional[PlasticityRule] = None):
        """
        領域間の投射を追加する。
        """
        if source not in self.neuron_groups or target not in self.neuron_groups:
            raise ValueError(f"Source {source} or Target {target} not found.")

        src_dim = self.neuron_groups[source].features  # type: ignore
        tgt_dim = self.neuron_groups[target].features  # type: ignore

        projection = SynapticProjection(src_dim, tgt_dim, plasticity_rule)
        self.projections[name] = projection.to(self.device)

        self.topology.append({"name": name, "src": source, "tgt": target})
        logger.debug(f"  + Projection added: {name} ({source} -> {target})")

    def forward_step(self, external_inputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        1タイムステップ分のシミュレーションを進める。
        kwargsには 'phase' (wake/sleep) などが含まれる。
        """
        self.time_step += 1

        # バッチサイズの推定と初期化
        batch_size = 1
        for inp in external_inputs.values():
            batch_size = inp.shape[0]
            break

        for name, group in self.neuron_groups.items():
            # type: ignore
            if self.prev_spikes[name] is None or self.prev_spikes[name].shape[0] != batch_size:
                num_neurons = group.features  # type: ignore
                self.prev_spikes[name] = torch.zeros(
                    batch_size, num_neurons, device=self.device)

        # 1. Integration (入力電流の計算)
        current_inputs: Dict[str, torch.Tensor] = {}

        # 内部結合からの入力
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module = self.projections[proj_name]
            src_spikes_prev = self.prev_spikes[src_name]

            # シナプス伝達
            synaptic_current = proj_module(src_spikes_prev)

            if tgt_name not in current_inputs:
                current_inputs[tgt_name] = synaptic_current
            else:
                current_inputs[tgt_name] = current_inputs[tgt_name] + \
                    synaptic_current

        # 外部入力の加算
        for group_name, inp in external_inputs.items():
            if group_name in self.neuron_groups:
                inp = inp.to(self.device)
                if group_name not in current_inputs:
                    current_inputs[group_name] = inp
                else:
                    # Shape check
                    if current_inputs[group_name].shape == inp.shape:
                        current_inputs[group_name] = current_inputs[group_name] + inp
                    else:
                        logger.warning(
                            f"Shape mismatch in input summation for {group_name}: {current_inputs[group_name].shape} vs {inp.shape}. Ignoring input.")

        # 2. Dynamics (ニューロン状態更新・発火)
        current_spikes: Dict[str, torch.Tensor] = {}

        for name, group in self.neuron_groups.items():
            if name in current_inputs:
                input_current = current_inputs[name]
            else:
                num_neurons = group.features  # type: ignore
                input_current = torch.zeros(
                    batch_size, num_neurons, device=self.device)

            # ニューロンモデルの実行
            spikes, _ = group(input_current)
            current_spikes[name] = spikes

        # 3. Plasticity (可塑性の適用)
        # ドキュメント通り、ここは「誤差逆伝播」ではなく「局所則」のみで更新される
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module = self.projections[proj_name]  # type: ignore

            if hasattr(proj_module, 'plasticity_rule') and proj_module.plasticity_rule is not None:
                src_spikes_prev = self.prev_spikes[src_name]
                tgt_spikes_curr = current_spikes[tgt_name]

                # ここで phase や dt などの情報を学習則に渡す
                proj_module.apply_plasticity(
                    pre_spikes=src_spikes_prev,
                    post_spikes=tgt_spikes_curr,
                    dt=self.dt,
                    **kwargs
                )

        # 4. Update State
        self.prev_spikes = current_spikes

        return {
            "spikes": current_spikes
        }

    def reset_state(self):
        """全ニューロンの状態リセット"""
        self.time_step = 0
        self.prev_spikes = {}

        for name, group in self.neuron_groups.items():
            if hasattr(group, 'reset'):
                group.reset()
            self.prev_spikes[name] = None

        logger.info("🔄 Substrate state reset.")


# --- Alias for Backward Compatibility ---
SNNCore = SpikingNeuralSubstrate
