# scripts/demos/run_dora_kernel_test.py
# Title: DORA Kernel Proof-of-Concept
# Description: 
#   行列演算なし(No-Matrix)、誤差逆伝播なし(No-BP)のイベント駆動カーネルの動作検証。
#   スパイクの伝播と、予測誤差によるシナプス可塑性(STDP + Predictive Error)を確認する。

import sys
import os
import random
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートにパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

from snn_research.hardware.event_driven_simulator import DORAKernel, SpikeEvent

def create_dummy_model():
    """
    DORAカーネルに読み込ませるための、ダミーのPyTorchモデル構造を作成。
    重みは初期化用に使われるだけで、実行時にはTensor演算は行われない。
    """
    model = nn.Sequential(
        nn.Linear(10, 20),  # 入力層(10) -> 隠れ層(20)
        nn.Linear(20, 5)    # 隠れ層(20) -> 出力層(5)
    )
    
    # 重みをランダムにスパース化（0に近い値を多くする）
    with torch.no_grad():
        for layer in model:
            if isinstance(layer, nn.Linear):
                # 80%の結合を0にする（スパース性）
                mask = torch.rand_like(layer.weight) > 0.8
                layer.weight.mul_(mask.float())
                # 残った結合の値を調整
                layer.weight.add_(torch.randn_like(layer.weight) * 0.5)
                
    return model

def run_simulation():
    print("⚡ initializing DORA Kernel (Event-Driven Mode)...")
    
    # 1. カーネルの初期化
    kernel = DORAKernel(dt=1.0)
    
    # 2. モデル構造のコンパイル (Torch -> Graph)
    torch_model = create_dummy_model()
    kernel.build_from_torch_model(torch_model)
    
    print(f"   Structure: {len(kernel.neurons)} neurons loaded.")
    
    # 3. 入力パターンの作成（繰り返し提示して学習させる）
    # パターンA: 0, 2, 4番目の入力ニューロンが同時に発火
    pattern_a_indices = [0, 2, 4]
    
    print("🚀 Starting Simulation Loop...")
    start_time = time.time()
    
    # 10回の試行（エポック）を行う
    for epoch in range(1, 11):
        current_time = epoch * 50.0 # 50msごとに刺激
        
        # 入力スパイクをスケジュール
        # パターンに少しノイズ（ジッター）を混ぜる
        jittered_time = current_time + random.uniform(0, 2.0)
        kernel.push_input_spikes(pattern_a_indices, jittered_time)
        
        # バックグラウンドノイズ（ランダムな発火）も少し入れる
        noise_indices = [random.randint(0, 9) for _ in range(2)]
        kernel.push_input_spikes(noise_indices, current_time + random.uniform(5, 10))

    # シミュレーション実行 (500ms)
    kernel.run(duration=600.0, learning_enabled=True)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    # --- 結果の表示 ---
    print("\n📊 Simulation Results:")
    print(f"   Execution Time: {elapsed:.4f} sec")
    print(f"   Total Spikes Processed: {kernel.stats['spikes']}")
    print(f"   Synaptic Ops (Add): {kernel.stats['ops']}")
    print(f"   Plasticity Events (Updates): {kernel.stats['plasticity_events']}")
    
    if kernel.stats['ops'] > 0:
        ops_per_sec = kernel.stats['ops'] / elapsed
        print(f"   Throughput: {ops_per_sec:.2f} OPS (Operations Per Second)")

    # --- 可視化: 発火ラスタプロット ---
    # カーネルからイベント履歴を取り出してプロットしたいが、
    # 簡易的に学習後の重みの変化を確認する
    
    print("\n🧠 Checking Plasticity (Weight Changes):")
    # 入力層(0-9)から隠れ層への結合で、Pattern A (0,2,4) に繋がる重みが強化されたか確認
    
    # 入力ニューロン0番から伸びるシナプスを調査
    n0 = kernel.neurons[0]
    strong_connections = [s for s in n0.outgoing_synapses if s.weight > 1.0]
    print(f"   Neuron 0 (Input) strong connections: {len(strong_connections)} synapses")
    for s in strong_connections:
        print(f"     -> To Neuron {s.target_id} (Weight: {s.weight:.2f})")
        
    # 入力ニューロン1番（刺激なし）から伸びるシナプス
    n1 = kernel.neurons[1]
    strong_connections_n1 = [s for s in n1.outgoing_synapses if s.weight > 1.0]
    print(f"   Neuron 1 (No Input) strong connections: {len(strong_connections_n1)} synapses")

if __name__ == "__main__":
    run_simulation()