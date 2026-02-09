# directory: tests/learning_rules
# file: test_rust_stdp_benchmark.py
# title: Rust STDP Benchmark
# description: Python版とRust版のSTDPの計算結果の一致を確認し、実行速度を比較します。

import torch
import time
import pytest
from snn_research.learning_rules.stdp import STDP

def test_rust_stdp_correctness():
    """Rust版とPython版の計算結果が一致することを確認"""
    print("\n\n--- Rust STDP Correctness Check ---")
    
    # 設定
    batch_size = 32
    input_dim = 128
    output_dim = 64
    
    # 共通の初期データを作成
    weights = torch.rand(output_dim, input_dim)
    pre_trace = torch.rand(batch_size, input_dim)
    post_trace = torch.rand(batch_size, output_dim)
    pre_spikes = (torch.rand(batch_size, input_dim) > 0.8).float()
    post_spikes = (torch.rand(batch_size, output_dim) > 0.8).float()
    
    stdp = STDP(learning_rate=0.01)
    
    # 1. PyTorch版の計算
    # 強制的にPyTorch版を呼ぶために内部メソッドを使用
    w_py = weights.clone()
    expected_w = stdp._update_pytorch(w_py, pre_spikes, post_spikes, pre_trace, post_trace, reward=None)
    
    # 2. Rust版の計算
    # CPUテンソルであれば自動的にRustが呼ばれるはずだが、明示的に確認
    w_rust = weights.clone()
    try:
        actual_w = stdp._update_rust(w_rust, pre_spikes, post_spikes, pre_trace, post_trace, lr=0.01)
    except Exception as e:
        pytest.fail(f"Rust execution failed: {e}")

    # 比較 (許容誤差 1e-5)
    diff = (expected_w - actual_w).abs().max()
    print(f"Max difference: {diff.item()}")
    
    assert torch.allclose(expected_w, actual_w, atol=1e-5), "Rust STDP result does not match PyTorch implementation!"
    print("✅ Correctness Test Passed!")

def test_rust_stdp_benchmark():
    """速度比較ベンチマーク"""
    print("\n--- Rust vs PyTorch Speed Benchmark ---")
    
    batch_size = 128
    input_dim = 1024 # 大きめの層
    output_dim = 1024
    iterations = 100
    
    weights = torch.rand(output_dim, input_dim)
    pre_trace = torch.rand(batch_size, input_dim)
    post_trace = torch.rand(batch_size, output_dim)
    pre_spikes = (torch.rand(batch_size, input_dim) > 0.9).float()
    post_spikes = (torch.rand(batch_size, output_dim) > 0.9).float()
    
    stdp = STDP()
    
    # PyTorch Benchmark
    start = time.time()
    for _ in range(iterations):
        stdp._update_pytorch(weights, pre_spikes, post_spikes, pre_trace, post_trace, reward=None)
    py_time = time.time() - start
    
    # Rust Benchmark
    # Warmup
    stdp._update_rust(weights, pre_spikes, post_spikes, pre_trace, post_trace, lr=0.01)
    
    start = time.time()
    for _ in range(iterations):
        stdp._update_rust(weights, pre_spikes, post_spikes, pre_trace, post_trace, lr=0.01)
    rust_time = time.time() - start
    
    print(f"PyTorch Time: {py_time:.4f}s")
    print(f"Rust Time:    {rust_time:.4f}s")
    print(f"Speedup:      {py_time / rust_time:.2f}x")

if __name__ == "__main__":
    test_rust_stdp_correctness()
    test_rust_stdp_benchmark()