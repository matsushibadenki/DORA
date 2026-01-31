# ファイルパス: snn_research/core/networks/bio_pc_network.py
# 日本語タイトル: Bio-Inspired Predictive Coding Network
# 目的・内容:
#   階層型予測符号化ネットワーク (Rao & Ballard, 1999 + SNN)。
#   各層は PredictiveCodingLayer で構成され、
#   入力画像に対する推論(状態推定)と生成(再構成)を同時に行う。

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, cast

from snn_research.core.network import AbstractNetwork
from snn_research.core.layers.predictive_coding import PredictiveCodingLayer
from snn_research.core.neurons.lif_neuron import LIFNeuron


class BioPCNetwork(AbstractNetwork):
    """
    生物学的予測符号化ネットワーク。
    Hierarchical Predictive Coding using SNN layers.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        layer_sizes: List[int],
        inference_steps: int = 10,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.layer_sizes = layer_sizes
        self.inference_steps = inference_steps

        # 入力次元の平坦化サイズ
        input_dim = int(torch.prod(torch.tensor(input_shape)).item())

        # レイヤー構築
        # Layers: [Input(L0)] <-> [L1] <-> [L2] ...
        # PCレイヤーは L1 から始まる (L0は入力層として扱う)
        self.pc_layers = nn.ModuleList()

        # L1: Input -> Hidden1
        # L1は下の層(Input)の予測誤差を受け取り、自身の状態を更新する
        l1 = PredictiveCodingLayer(
            input_size=input_dim,
            hidden_size=layer_sizes[0],
            neuron_class=LIFNeuron,
            neuron_params=kwargs.get("neuron_params", {}),
            inference_steps=inference_steps,
            learning=True,
        )
        self.pc_layers.append(l1)

        # L2...Ln
        for i in range(1, len(layer_sizes)):
            prev_hidden = layer_sizes[i - 1]
            curr_hidden = layer_sizes[i]

            l = PredictiveCodingLayer(
                input_size=prev_hidden,
                hidden_size=curr_hidden,
                neuron_class=LIFNeuron,
                neuron_params=kwargs.get("neuron_params", {}),
                inference_steps=inference_steps,
                learning=True,
            )
            self.pc_layers.append(l)

        self.built = True

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        推論と学習のステップを実行。

        Args:
            x: Sensory Input [Batch, InputDim]

        Returns:
            Dict: {
                'reconstruction': Input層へのトップダウン予測,
                'states': 各層の状態,
                'errors': 各層の予測誤差
            }
        """
        batch_size = x.shape[0]

        # 1. 状態の初期化 (または前回の状態の継承)
        # 本来は時間相関を利用するため前回の状態を保持すべきだが、
        # ここでは簡易化のためゼロ初期化、もしくはステートフルにするならメンバ変数を使う
        states: List[torch.Tensor] = []
        for size in self.layer_sizes:
            states.append(torch.zeros(batch_size, size, device=x.device))

        # 最上位層へのトップダウン入力（事前分布、通常は0またはノイズ）
        top_prior = torch.zeros(batch_size, self.layer_sizes[-1], device=x.device)

        # 2. 推論 (Inference Phase) - Iterative Relaxation
        # PredictiveCodingLayer.forward は内部で relaxation loop を回す設計になっているため、
        # ここでは層ごとの結合を管理する。
        # ただし、層間の相互作用が必要なため、ネットワーク全体でループを回すのが一般的。
        # 今回の PredictiveCodingLayer は「ボトムアップ入力」と「トップダウン状態」を受け取る。

        # 層ごとの入出力を保持
        layer_errors: List[torch.Tensor] = [torch.zeros(1)] * len(self.pc_layers)
        layer_spikes: List[torch.Tensor] = [torch.zeros(1)] * len(self.pc_layers)

        # 入力層のデータ
        current_bottom_input = x.view(batch_size, -1)

        # 双方向の信号伝播
        # 下から上へ (Bottom-Up Error Pass) と 上から下へ (Top-Down Prediction Pass)
        # を収束するまで繰り返すが、PC Layer自体がStepsを持っているので、
        # ここでは単純に層を順番に実行する（簡易実装）。
        # ※本来は全体最適化が必要

        for i, layer in enumerate(self.pc_layers):
            layer_module = cast(PredictiveCodingLayer, layer)

            # 上位層からの状態（最上位ならPrior）
            if i == len(self.pc_layers) - 1:
                top_state = top_prior
                top_error = None
            else:
                top_state = states[
                    i + 1
                ]  # 次の層の状態がトップダウン入力になる（※初期化時は0）
                top_error = None  # 簡易化

            # レイヤー実行 (Relaxation)
            # bottom_up_input = 下層からの信号（L1の場合は画像、L2以降は下層の状態/誤差）
            # ここでは「下層の誤差」ではなく「下層の状態」を入力として受け取り、内部で誤差計算する場合と設計が分かれる。
            # PredictiveCodingLayerの実装を見ると:
            #   raw_error = bottom_up_input - pred
            # となっているので、bottom_up_input は「ターゲット信号」である。

            # L1への入力は画像(x)。
            # L2への入力はL1の状態(states[0])。
            target_signal = current_bottom_input

            updated_state, final_error, _, spikes = layer_module(
                bottom_up_input=target_signal,
                top_down_state=states[i],  # 自身の現在の状態（初期値）
                top_down_error=top_error,
            )

            # 状態更新
            states[i] = updated_state
            layer_errors[i] = final_error
            layer_spikes[i] = spikes

            # 次の層へのターゲット信号は、この層の状態(State)となる
            current_bottom_input = (
                updated_state.detach()
            )  # 勾配を切るか繋ぐかは学習戦略による

        # 3. 学習 (Weight Update) - Local STDP
        if self.training:
            for i, layer in enumerate(self.pc_layers):
                layer_module = cast(PredictiveCodingLayer, layer)

                # Generative Connection: Hidden (Pre) -> Input (Post)
                # Pre: Current Layer State (Hidden)
                # Use states[i] (Current Hidden)
                # Note: states[i] might be analog, convert to spikes/rate proxy
                pre_spikes = (states[i].abs() > 0.01).float()

                # Post: Lower Layer State (or Input Image)
                if i == 0:
                    # L1 Input is the original image x.
                    post_spikes = (x.view(batch_size, -1).abs() > 0.1).float()
                else:
                    # Lower Layer Hidden State
                    post_spikes = (states[i - 1].abs() > 0.01).float()

                # 重み更新
                layer_module.update_weights(
                    bottom_input=None,
                    top_state=pre_spikes,
                    error=layer_errors[i],
                    spikes=post_spikes,
                )

        # L1の再構成画像 (Prediction = Generative(State))
        # 再構成を取得するにはGenerative Pathを通す必要がある
        l1 = cast(PredictiveCodingLayer, self.pc_layers[0])
        # norm_state -> generative_fc -> generative_neuron -> output
        # ここでは簡易的に線形変換のみで再構成近似を取得
        with torch.no_grad():
            recon = l1.generative_fc(l1.norm_state(states[0]))

        return {
            "reconstruction": recon,
            "states": torch.cat([s.view(batch_size, -1) for s in states], dim=1),
            "errors": torch.cat([e.view(batch_size, -1) for e in layer_errors], dim=1),
        }

    def reset_state(self) -> None:
        """状態リセット"""
        # PredictiveCodingLayerはステートフルではない(forwardで完結する)設計に寄せているが、
        # もし内部状態を持つならここでリセット
        pass
