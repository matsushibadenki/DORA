# directory: scripts/visualization
# file: visualize_sara_dynamics.py
# purpose: SARAエンジンの内部ダイナミクス（スパイク、メモリ、思考軌跡）の可視化

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from torchvision import datasets, transforms

# パス設定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from snn_research.models.experimental.sara_engine import SARAEngine
except ImportError:
    sys.path.append(".")
    from snn_research.models.experimental.sara_engine import SARAEngine

def visualize_dynamics(model_path):
    device = torch.device("cpu") # 可視化はCPUで十分
    
    # 1. モデルのロード
    model = SARAEngine(
        input_dim=784,
        n_encode_neurons=128,
        d_legendre=64,
        d_meaning=128,
        n_output=10
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print("Model file not found. Please run training first.")
        return

    model.eval()

    # 2. データ取得（テストデータから数枚）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=False, download=True, transform=transform)
    
    # 特定の数字（例: 0, 1, 2）をピックアップ
    targets = [0, 1, 2, 8]
    samples = []
    labels = []
    
    for img, label in dataset:
        if label in targets and label not in labels:
            samples.append(img)
            labels.append(label)
        if len(samples) == len(targets):
            break
            
    # 3. 推論と内部状態の記録
    # モデルのforwardメソッドを少しハックして内部状態を取り出すか、
    # 個別にモジュールを呼び出す
    
    fig, axes = plt.subplots(len(samples), 4, figsize=(20, 4 * len(samples)))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    for i, (x_img, label) in enumerate(zip(samples, labels)):
        x_flat = x_img.view(1, -1).to(device)
        
        with torch.no_grad():
            # (A) Encoder: Spikes
            spikes, _ = model.encoder(x_flat) # (1, T, N)
            
            # (B) Memory: Legendre State
            m = model.memory(spikes) # (1, d_legendre)
            
            # (C) Reasoning: Trajectory
            # RLMの内部ループを手動で回して軌跡を取得
            h = torch.zeros(1, model.rlm.cell.hidden_size).to(device)
            trajectory = []
            max_depth = 10
            
            # 初期注入
            h = model.rlm.cell(m, h)
            trajectory.append(h.numpy().flatten())
            
            for _ in range(max_depth - 1):
                h = model.rlm.cell(m, h)
                trajectory.append(h.numpy().flatten())
            
            # (D) Decoder output
            z = model.rlm.ln(h)
            logits = model.decoder(z)
            probs = F.softmax(logits, dim=1)
            pred = probs.argmax().item()

        # --- Plotting ---
        
        # 1. Input Image
        ax_img = axes[i, 0]
        ax_img.imshow(x_img.squeeze(), cmap='gray')
        ax_img.set_title(f"Input: {label} (Pred: {pred})")
        ax_img.axis('off')
        
        # 2. Spike Raster Plot
        ax_spike = axes[i, 1]
        spike_data = spikes.squeeze().numpy().T # (N, T)
        ax_spike.imshow(spike_data, aspect='auto', cmap='binary', interpolation='nearest')
        ax_spike.set_title("SNN Spike Raster (Time vs Neurons)")
        ax_spike.set_xlabel("Time Step")
        ax_spike.set_ylabel("Neuron ID")
        
        # 3. Legendre Memory State
        ax_mem = axes[i, 2]
        mem_data = m.numpy().T # (d, 1)
        ax_mem.bar(range(len(mem_data)), mem_data.flatten())
        ax_mem.set_title("Legendre Memory State")
        ax_mem.set_xlabel("Dimension")
        ax_mem.set_ylim(-2, 2)
        
        # 4. Thinking Trajectory (PCA projection of Reasoning)
        ax_traj = axes[i, 3]
        traj_data = np.array(trajectory) # (Depth, Hidden)
        
        # 簡易的に最初の2次元を表示（またはノルムの変化）
        # PCAが使えない環境も考慮して、変化量(Velocity)をプロット
        # 思考が収束していく様子（変化量が減る）を確認
        diffs = np.linalg.norm(np.diff(traj_data, axis=0), axis=1)
        ax_traj.plot(diffs, marker='o', linestyle='-')
        ax_traj.set_title("Thinking Convergence (State Delta)")
        ax_traj.set_xlabel("Recursion Depth")
        ax_traj.set_ylabel("Change in State ||z_t - z_{t-1}||")
        ax_traj.grid(True)
        
    output_file = "workspace/runtime_state/sara_dynamics_visualization.png"
    plt.savefig(output_file)
    print(f"\nVisualization saved to {output_file}")
    print("Please inspect the 'Thinking Convergence' plot.")
    print(" - Rapid decrease means fast intuitive thinking.")
    print(" - Slow decrease or oscillation implies deep contemplation.")

if __name__ == "__main__":
    visualize_dynamics("models/checkpoints/sara_mnist_v5.pth")