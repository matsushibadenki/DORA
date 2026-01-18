"""
Latency Benchmark Script
Benchmarks the latency of the SNNCore model.
Checks for available devices (MPS/CUDA/CPU) automatically.
"""

import logging
import os
import sys
import time
from typing import Any, Dict, cast

import torch

# プロジェクトルートをパスに追加
sys.path.append(os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))
from snn_research.core.snn_core import SNNCore


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """利用可能な最適なデバイスを取得します (MPS > CUDA > CPU)。"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def benchmark() -> None:
    print("⏱️ Starting Latency Benchmark...")

    device = get_device()
    print(f"Running on Device: {device}")

    # Mock config object
    # SNNCoreのデフォルト in_features は 128 です。
    model_conf = {
        "hidden_dim": 128,
        "layers": 2
    }

    try:
        # [Mypy Fix] config引数に対して明示的にDict[str, Any]へキャスト
        # SNNCoreの初期化 (SpikingNeuralSubstrate v3.3 API準拠)
        model = SNNCore(
            config=cast(Dict[str, Any], model_conf),
            device=device
        )
        # ネットワーク構築
        # 1. ニューロン層の追加
        # Input Layer (Encoder代わり)
        model.add_neuron_group("Input", 128)
        # Hidden Layer
        model.add_neuron_group("Hidden", 128)
        # Output Layer
        model.add_neuron_group("Output", 10)

        # 2. 投射 (Synapses) の追加
        model.add_projection("Input->Hidden", "Input", "Hidden")
        model.add_projection("Hidden->Output", "Hidden", "Output")

        model.to(device)
        model.eval()  # ベンチマーク時は評価モード推奨

        # 入力テンソルの作成
        # バッチサイズ 1, 特徴量 128
        input_tensor = torch.randn(1, 128).to(device)

        # Warmup (デバイスキャッシュの初期化)
        print("Warmup...")
        with torch.no_grad():
            for _ in range(10):
                # forward_step は dict を受け取り dict を返す
                _ = model.forward_step({"Input": input_tensor})

        # Measurement
        print("Measuring...")
        latencies = []
        with torch.no_grad():
            for _ in range(100):
                start = time.perf_counter()

                # 1ステップ実行
                _ = model.forward_step({"Input": input_tensor})

                torch.cuda.synchronize() if device.type == "cuda" else None  # CUDAの場合の同期
                # MPSの場合は同期が自動的に行われることが多いが、必要に応じて torch.mps.synchronize() を追加可能
                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        avg_latency = sum(latencies) / len(latencies)
        print(f"Average Latency: {avg_latency:.4f} ms")

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark()
