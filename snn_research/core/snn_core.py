# ファイルパス: snn_research/core/snn_core.py
# 日本語タイトル: Spiking Neural Substrate (The Kernel) Refactored
# 目的・内容:
#   Neuromorphic OSの中核となる神経基盤クラス。
#   ニューロン集団(Groups)とシナプス結合(Projections)の管理、および
#   物理シミュレーション（統合・発火・可塑性）のステップ実行を行う。

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
from torch import Tensor

# 絶対インポートを使用（パッケージ構造が前提）
from snn_research.core.neurons.lif_neuron import LIFNeuron
from snn_research.learning_rules.base_rule import PlasticityRule

logger = logging.getLogger(__name__)


class SynapticProjection(nn.Module):
    """
    ニューロン集団間のシナプス結合を表現するクラス。
    学習則（PlasticityRule）を保持し、ローカルな更新を適用する。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        plasticity_rule: Optional[PlasticityRule] = None
    ) -> None:
        super().__init__()
        # バイアスなしの線形層としてシナプスをモデル化
        self.synapse = nn.Linear(in_features, out_features, bias=False)
        self.plasticity_rule = plasticity_rule

        # 直交初期化による信号伝播の安定化
        nn.init.orthogonal_(self.synapse.weight, gain=1.4)

    def forward(self, x: Tensor) -> Tensor:
        """入力スパイク列に対するシナプス電流を計算"""
        return self.synapse(x)

    def apply_plasticity(
        self,
        pre_spikes: Tensor,
        post_spikes: Tensor,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """学習則の適用"""
        if self.plasticity_rule is None:
            return {}

        logs: Dict[str, Any] = {}
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
                # 重みの発散を防ぐためのクリッピング（安定性確保）
                self.synapse.weight.data.clamp_(-5.0, 5.0)

        return logs


class SpikingNeuralSubstrate(nn.Module):
    """
    Neuromorphic OSのための汎用神経基盤（Kernel）。
    明示的な時間ステップまたはイベント駆動で状態を更新する。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device = torch.device('cpu'),
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.config = config
        self.device = device

        self.time_step: int = 0
        self.dt: float = config.get("dt", 1.0)

        self.neuron_groups: nn.ModuleDict = nn.ModuleDict()
        self.projections: nn.ModuleDict = nn.ModuleDict()
        # トポロジー情報: {'name': str, 'src': str, 'tgt': str}
        self.topology: List[Dict[str, str]] = []

        # 前回のスパイク状態（STDP等で使用）: {group_name: Tensor}
        self.prev_spikes: Dict[str, Optional[Tensor]] = {}

        logger.info("⚡ SpikingNeuralSubstrate initialized.")

    def add_neuron_group(
        self,
        name: str,
        num_neurons: int,
        neuron_model: Optional[nn.Module] = None
    ) -> None:
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
            # mypy用キャスト: set_statefulメソッドを持つと仮定
            cast(Any, neuron_model).set_stateful(True)

        self.neuron_groups[name] = neuron_model.to(self.device)
        self.prev_spikes[name] = None

        logger.debug(f"  + Group added: {name} ({num_neurons} neurons)")

    def add_projection(
        self,
        name: str,
        source: str,
        target: str,
        plasticity_rule: Optional[PlasticityRule] = None
    ) -> None:
        """
        領域間の投射を追加する。
        """
        if source not in self.neuron_groups or target not in self.neuron_groups:
            raise ValueError(f"Source {source} or Target {target} not found.")

        # ModuleDictから取り出す際はnn.Module型なので、属性アクセス用にキャストが必要な場合があるが、
        # ここではfeatures属性を持っていると仮定して取得
        src_module = cast(Any, self.neuron_groups[source])
        tgt_module = cast(Any, self.neuron_groups[target])

        src_dim = int(src_module.features)
        tgt_dim = int(tgt_module.features)

        projection = SynapticProjection(src_dim, tgt_dim, plasticity_rule)
        self.projections[name] = projection.to(self.device)

        self.topology.append({"name": name, "src": source, "tgt": target})
        logger.debug(f"  + Projection added: {name} ({source} -> {target})")

    def get_firing_rates(self) -> Dict[str, float]:
        """
        各ニューロン集団の平均発火率（直近ステップ）を返す。
        """
        rates: Dict[str, float] = {}
        for name, spikes in self.prev_spikes.items():
            if spikes is not None:
                rates[name] = float(spikes.mean().item())
            else:
                rates[name] = 0.0
        return rates

    def forward_step(
        self,
        external_inputs: Dict[str, Tensor],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        1タイムステップ分のシミュレーションを進める。
        """
        self.time_step += 1

        # バッチサイズの推定
        batch_size = 1
        for inp in external_inputs.values():
            batch_size = inp.shape[0]
            break

        # 前回のスパイク状態の初期化（未定義の場合）
        self._initialize_prev_spikes_if_needed(batch_size)

        # 1. Integration: シナプス電流と外部入力の統合
        current_inputs = self._integrate_inputs(external_inputs, batch_size)

        # 2. Dynamics: ニューロン状態更新・発火
        current_spikes = self._update_neuron_dynamics(
            current_inputs, batch_size
        )

        # 3. Plasticity: 可塑性ルールの適用
        self._apply_plasticity(current_spikes, **kwargs)

        # 4. Update State: 状態の保存
        # current_spikesの値はTensorであり、Optional[Tensor]に適合する
        self.prev_spikes = cast(Dict[str, Optional[Tensor]], current_spikes)

        return {
            "spikes": current_spikes
        }

    def _initialize_prev_spikes_if_needed(self, batch_size: int) -> None:
        """バッチサイズに合わせて前回のスパイクバッファを初期化"""
        for name, group in self.neuron_groups.items():
            prev = self.prev_spikes.get(name)
            if prev is None or prev.shape[0] != batch_size:
                group_module = cast(Any, group)
                num_neurons = int(group_module.features)
                self.prev_spikes[name] = torch.zeros(
                    batch_size, num_neurons, device=self.device
                )

    def _integrate_inputs(
        self,
        external_inputs: Dict[str, Tensor],
        batch_size: int
    ) -> Dict[str, Tensor]:
        """
        内部結合と外部入力を統合して、各ニューロン集団への入力電流を計算する。
        """
        current_inputs: Dict[str, Tensor] = {}

        # 内部結合（再帰・フィードフォワード・フィードバック）からの入力
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module = self.projections[proj_name]
            # prev_spikesは初期化済みであることが保証されているため cast
            src_spikes_prev = cast(Tensor, self.prev_spikes[src_name])

            # シナプス伝達
            synaptic_current = proj_module(src_spikes_prev)

            if tgt_name not in current_inputs:
                current_inputs[tgt_name] = synaptic_current
            else:
                current_inputs[tgt_name] = current_inputs[tgt_name] + synaptic_current

        # 外部入力の加算
        for group_name, inp in external_inputs.items():
            if group_name not in self.neuron_groups:
                continue

            inp = inp.to(self.device)
            if group_name not in current_inputs:
                current_inputs[group_name] = inp
            else:
                if current_inputs[group_name].shape == inp.shape:
                    current_inputs[group_name] = current_inputs[group_name] + inp
                else:
                    logger.warning(
                        f"Shape mismatch in input summation for {group_name}: "
                        f"{current_inputs[group_name].shape} vs {inp.shape}. "
                        "Ignoring external input."
                    )
        
        return current_inputs

    def _update_neuron_dynamics(
        self,
        current_inputs: Dict[str, Tensor],
        batch_size: int
    ) -> Dict[str, Tensor]:
        """
        各ニューロン集団の状態を更新し、スパイクを生成する。
        """
        current_spikes: Dict[str, Tensor] = {}

        for name, group in self.neuron_groups.items():
            if name in current_inputs:
                input_current = current_inputs[name]
            else:
                group_module = cast(Any, group)
                num_neurons = int(group_module.features)
                input_current = torch.zeros(
                    batch_size, num_neurons, device=self.device
                )

            # ニューロンモデルの実行 (forward)
            # 多くのモデルは (spikes, state) を返すが、ここでは spikes のみを使用
            spikes, _ = group(input_current)
            current_spikes[name] = spikes

        return current_spikes

    def _apply_plasticity(
        self,
        current_spikes: Dict[str, Tensor],
        **kwargs: Any
    ) -> None:
        """
        トポロジーに基づいて可塑性ルールを適用する。
        """
        for conn in self.topology:
            proj_name = conn['name']
            src_name = conn['src']
            tgt_name = conn['tgt']

            proj_module_plastic = cast(Any, self.projections[proj_name])

            # 学習則が設定されている場合のみ計算
            if (hasattr(proj_module_plastic, 'plasticity_rule') and 
                    proj_module_plastic.plasticity_rule is not None):
                
                src_spikes_prev = cast(Tensor, self.prev_spikes[src_name])
                tgt_spikes_curr = current_spikes[tgt_name]

                proj_module_plastic.apply_plasticity(
                    pre_spikes=src_spikes_prev,
                    post_spikes=tgt_spikes_curr,
                    dt=self.dt,
                    **kwargs
                )

    def reset_state(self) -> None:
        """全ニューロンの状態リセット"""
        self.time_step = 0
        self.prev_spikes = {}

        for name, group in self.neuron_groups.items():
            if hasattr(group, 'reset'):
                # mypy: resetメソッドがあると仮定
                cast(Any, group).reset()
            self.prev_spikes[name] = None

        logger.info("🔄 Substrate state reset.")

    def get_total_spikes(self) -> int:
        """全ニューロンのスパイク総数を返す（統計用）"""
        total = 0
        for spikes in self.prev_spikes.values():
            if spikes is not None:
                total += int(spikes.sum().item())
        return total


# --- Backward Compatibility Alias ---
SNNCore = SpikingNeuralSubstrate