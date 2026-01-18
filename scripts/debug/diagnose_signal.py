# isort: skip_file
from omegaconf import OmegaConf
import torch
import sys
import os
from pathlib import Path

# プロジェクトルートをパスに追加 (インポートの前に行う)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


# 必要なライブラリのインポート (sys.path設定後)
try:
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from snn_research.data.datasets import SimpleTextDataset
    # [FIX] Use high-level model instead of Kernel
    # from snn_research.core.snn_core import SNNCore
    from snn_research.models.transformer.spiking_rwkv import BitSpikingRWKV
except ImportError as e:
    print(f"❌ ライブラリのインポートに失敗しました: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


def main():
    print("🔍 SNN詳細信号診断 (Full Forward / Fixed) を開始します...")

    # 1. 設定とモデル
    # scripts/debug/../../configs -> project_root/configs
    config_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))), "configs/models/bit_rwkv_micro.yaml")

    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return

    cfg = OmegaConf.load(config_path)

    print("  - Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Padding token setting for GPT-2
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        return

    device = "cpu"

    print("  - Building model...")
    try:
        # [FIX] Instantiate BitSpikingRWKV directly
        # Config structure might differ, map accordingly or pass flat args if model doesn't take config obj
        # BitSpikingRWKV(vocab_size, d_model=..., num_layers=...)

        mdl_cfg = cfg.model

        # Extract params safely or use defaults
        d_model = mdl_cfg.get("d_model", 256)
        num_layers = mdl_cfg.get("num_layers", 4)
        time_steps = mdl_cfg.get("time_steps", 16)

        # If config is nested differently, adjust here.

        model = BitSpikingRWKV(
            vocab_size=len(tokenizer),
            d_model=d_model,
            num_layers=num_layers,
            time_steps=time_steps,
            # neuron_config can be passed if needed
        )

        model.to(device)
        model.eval()
    except Exception as e:
        print(f"❌ モデル構築エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # 2. データ準備
    data_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__))), "data/smoke_test_data.jsonl")

    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        # Create dummy data if missing for diagnosis
        print("⚠️  Using dummy data instead.")
        dummy_data = True
    else:
        dummy_data = False

    if dummy_data:
        # Create minimal dummy dataset interface
        class DummyDataset:
            def __len__(self): return 1
            def __getitem__(self, idx): return (torch.randint(
                0, 1000, (16,)), torch.randint(0, 1000, (16,)))
        dataset = DummyDataset()
    else:
        dataset = SimpleTextDataset(data_path, tokenizer, max_seq_len=16)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    try:
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)):
            input_ids = batch[0].to(device)
        elif isinstance(batch, dict):
            input_ids = batch['input_ids'].to(device)
        elif isinstance(batch, torch.Tensor):
            input_ids = batch.to(device)
        else:
            print(f"❌ Unexpected batch type: {type(batch)}")
            return
    except StopIteration:
        print("❌ データセットが空です。")
        return

    # 3. 詳細診断実行
    print("\n📊 レイヤー別信号追跡:")

    # フック関数: 入出力の統計を表示 (修正: floatキャスト追加)
    def debug_hook(name):
        def hook(module, input, output):
            if isinstance(input, tuple):
                input = input[0]
            if isinstance(output, tuple):
                output = output[0]

            # float()にキャストしてから計算することでエラーを回避
            in_mean = input.float().abs().mean().item(
            ) if isinstance(input, torch.Tensor) else 0.0
            out_mean = output.float().abs().mean().item(
            ) if isinstance(output, torch.Tensor) else 0.0
            out_max = output.float().abs().max().item(
            ) if isinstance(output, torch.Tensor) else 0.0

            # スパイク数 (ニューロンの場合)
            spike_info = ""
            if "lif" in name.lower() or "neuron" in name.lower():
                if isinstance(output, torch.Tensor):
                    spike_count = output.sum().item()
                    # Only calculate rate if count > 0 to avoid noise
                    if spike_count > 0:
                        spike_rate = output.float().mean().item() * 100
                        spike_info = f" | Spikes: {int(spike_count)} (Rate: {spike_rate:.2f}%)"
                    else:
                        spike_info = " | Spikes: 0"

            print(f"  🔹 [{name}]")
            print(f"      Input Mean: {in_mean:.6f}")
            print(
                f"      Output Mean: {out_mean:.6f} | Max: {out_max:.6f}{spike_info}")

            if out_max == 0 and "lif" not in name.lower() and "neuron" not in name.lower():
                # Embedding入力(Long)は除く
                if "embedding" not in name.lower():
                    print(f"      🚨 信号消失警報: {name} の出力がゼロです！")
        return hook

    # 主要な層にフックを登録
    hooks = []

    # Embedding - BitSpikingRWKV has 'embedding' ? Let's check or user generic try
    if hasattr(model, 'embedding'):
        hooks.append(model.embedding.register_forward_hook(
            debug_hook("Embedding")))

    # Layers
    if hasattr(model, 'blocks'):  # RWKV usually has blocks
        for i, layer in enumerate(model.blocks):
            # Inspect structure if needed
            hooks.append(layer.register_forward_hook(debug_hook(f"Block{i}")))

            # BitLinear if present
            if hasattr(layer, 'time_mixing'):
                # Check submodules
                pass

    # 実行
    with torch.no_grad():
        try:
            print("\n  --- Forward Pass Start ---")
            # RWKV forward sig: forward(self, input_ids, return_spikes=False, ...)
            model(input_ids, return_spikes=True)
            print("  --- Forward Pass End ---")
        except Exception as e:
            print(f"❌ 実行エラー: {e}")
            import traceback
            traceback.print_exc()

    # 後始末
    for h in hooks:
        h.remove()
    print("\n✅ 診断終了")


if __name__ == "__main__":
    main()
