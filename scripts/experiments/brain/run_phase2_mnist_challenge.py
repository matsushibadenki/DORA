# ファイルパス: scripts/experiments/brain/run_phase2_mnist_challenge.py
import sys
import os
import time
import logging
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import numpy as np

# パス解決
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from snn_research.models.visual_cortex import VisualCortex as VisualCortexV2

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase2_MNIST")

class MNISTOverlayProcessor:
    """
    画像にラベル情報を埋め込む（Supervised Forward-Forward用）
    画像(784) + ワンホットラベル(10) = 794次元入力
    """
    def __init__(self, device):
        self.device = device

    def overlay_label(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28) or (B, 784)
        labels: (B,)
        Returns: (B, 794)
        """
        x = x.view(x.size(0), -1).to(self.device)
        labels = labels.to(self.device)

        # 画像の正規化 (0-1 -> 0.0-1.0)
        x = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)

        # ワンホットラベルの生成
        one_hot = F.one_hot(labels, num_classes=10).float()
        # ラベル信号を少し強めにする（初期学習のガイド用）
        one_hot = one_hot * 1.5 

        # 結合
        return torch.cat([x, one_hot], dim=1)

def get_mnist_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # データセットのダウンロード先を workspace/data に指定
    data_path = "workspace/data"
    os.makedirs(data_path, exist_ok=True)
    
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def evaluate(brain, test_loader, processor, device):
    brain.eval() # 評価モード（ノイズなし）
    correct = 0
    total = 0
    
    logger.info("🔍 Evaluating...")
    
    # 評価は時間がかかるので最初の1000枚だけチェック
    limit_samples = 1000
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if total >= limit_samples: break
            
            data = data.to(device)
            target = target.to(device)
            batch_size = data.size(0)
            
            # 各クラスのラベルを埋め込んでGoodnessを計測
            # (Batch, 10, 794) のテンソルを作って一括処理したいが、
            # SNNの状態リセットが必要なため、シンプルにループで回す（精度優先）
            
            # 予測ラベル格納用
            batch_goodness = []
            
            for label_idx in range(10):
                brain.reset_state()
                
                # 全員に同じ label_idx を埋め込む
                dummy_labels = torch.full((batch_size,), label_idx, dtype=torch.long, device=device)
                x_in = processor.overlay_label(data, dummy_labels)
                
                # 推論実行
                brain(x_in, phase="inference")
                stats = brain.get_goodness()
                
                # 全層のGoodnessを合算してスコアとする
                # 特に深層(V3)の反応を重視
                score = stats.get("V2_goodness", 0) + stats.get("V3_goodness", 0) * 2.0
                batch_goodness.append(score)
            
            # batch_goodness: List of scalars (これはバッチ処理できていない簡易実装)
            # 正しくはバッチ内の個々のサンプルごとのGoodnessを見る必要がある。
            # SNNの current implementation の get_goodness() は mean() を返してしまうため、
            # バッチサイズ=1 で評価するか、get_goodnessを改修する必要がある。
            # 今回は「バッチサイズ=1」で正確に評価する形に変更する。
            pass

    # --- バッチサイズ1での正確な評価ループ ---
    correct = 0
    total = 0
    
    # テストローダーを再作成 (Batch=1)
    test_loader_single = DataLoader(test_loader.dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader_single):
            if i >= 100: break # 時間短縮のため100枚で速報値を出す
            
            data = data.to(device)
            target = target.item()
            
            best_goodness = -1.0
            predicted_label = -1
            
            for label_c in range(10):
                brain.reset_state()
                
                # 候補ラベルを埋め込む
                lbl = torch.tensor([label_c], device=device)
                x_in = processor.overlay_label(data, lbl)
                
                # 推論
                brain(x_in, phase="inference")
                
                # Goodness取得 (Batch=1なのでスカラーでOK)
                stats = brain.get_goodness()
                g = stats.get("V2_goodness", 0) + stats.get("V3_goodness", 0)
                
                if g > best_goodness:
                    best_goodness = g
                    predicted_label = label_c
            
            if predicted_label == target:
                correct += 1
            total += 1
            
            if i % 20 == 0:
                print(f".", end="", flush=True)

    print()
    acc = 100.0 * correct / total
    return acc

def run_mnist_challenge():
    logger.info("🧠 Starting Phase 2 MNIST Challenge (Backprop FREE)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # 1. コンフィグ設定
    # 入力次元 = 784(画像) + 10(ラベル) = 794
    config = {
        "input_dim": 794, 
        "hidden_dim": 1500, # 容量を少し増やす
        "num_layers": 3,
        "dt": 1.0,
        "tau_mem": 5.0,
        "learning_rate": 0.08, # 学習率調整
        "ff_threshold": 3.0,
        "noise_level": 1.0
    }
    
    # 2. モデルとデータの準備
    brain = VisualCortexV2(device, config).to(device)
    processor = MNISTOverlayProcessor(device)
    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    
    logger.info(f"Brain Initialized. Input Dim: {config['input_dim']}")
    
    # 3. 学習ループ
    epochs = 2 # SNNなのでエポック数は少なめで様子見
    
    for epoch in range(1, epochs + 1):
        logger.info(f"Epoch {epoch}/{epochs} Start")
        brain.train()
        
        start_time = time.time()
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            # --- Positive Pass (Wake: 正解ラベル) ---
            brain.reset_state()
            x_pos = processor.overlay_label(data, target)
            brain(x_pos, phase="wake")
            
            # --- Negative Pass (Sleep/Dream: 誤りラベル) ---
            brain.reset_state()
            # ランダムな誤りラベルを生成
            rnd_labels = torch.randint(0, 10, target.shape, device=device)
            # 正解と同じになってしまったものは +1 してずらす
            rnd_labels = torch.where(rnd_labels == target, (rnd_labels + 1) % 10, rnd_labels)
            
            x_neg = processor.overlay_label(data, rnd_labels)
            brain(x_neg, phase="sleep")
            
            batch_count += 1
            if batch_count % 100 == 0:
                metrics = brain.get_stability_metrics()
                v1_rate = metrics.get("V1_firing_rate", 0)
                v3_rate = metrics.get("V3_firing_rate", 0)
                logger.info(f"  Batch {batch_count}: V1 Rate={v1_rate:.1%} V3 Rate={v3_rate:.1%}")

        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch} Finished in {epoch_time:.1f}s")
        
        # 4. 途中評価
        acc = evaluate(brain, test_loader, processor, device)
        logger.info(f"📊 Epoch {epoch} Test Accuracy: {acc:.2f}%")
        
        # 安定性チェック
        if acc < 15.0 and epoch > 1:
            logger.warning("⚠️ Accuracy is low. Learning might be unstable.")
    
    # 最終レポート保存
    with open("workspace/reports/mnist_result.txt", "w") as f:
        f.write(f"MNIST Accuracy: {acc}%\n")

if __name__ == "__main__":
    run_mnist_challenge()