# ファイルパス: snn_research/visualization/spike_plotter.py
# 日本語タイトル: Spike Activity Plotter
# 目的・内容:
#   - 脳内のスパイク活動を可視化画像(Numpy Array)に変換する。
#   - Gradioで表示可能な形式で出力する。

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUIバックエンドを使わない設定
import matplotlib.pyplot as plt
import io
from PIL import Image
from typing import Dict, Any

class SpikePlotter:
    """
    ニューロン発火のラスタープロットやヒートマップを生成するクラス。
    """
    @staticmethod
    def plot_substrate_state(substrate_state: Dict[str, Any]) -> np.ndarray:
        """
        現在の基盤状態(spikes)を受け取り、領域ごとの発火状況を画像化して返す。
        """
        spikes_dict = substrate_state.get("spikes", {})
        
        # プロットの準備
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # 各領域のスパイクを結合して1つのベクトルにする
        combined_spikes = []
        labels = []
        boundaries = [0]
        
        for name, spikes in spikes_dict.items():
            if spikes is not None:
                # (Batch, Neurons) -> (Neurons)
                flat_spikes = spikes.detach().cpu().numpy().flatten()
                combined_spikes.append(flat_spikes)
                labels.append(name)
                boundaries.append(boundaries[-1] + len(flat_spikes))
        
        if not combined_spikes:
            plt.close(fig)
            return np.zeros((100, 400, 3), dtype=np.uint8)

        all_spikes = np.concatenate(combined_spikes)
        
        # バーコードのような可視化 (1D heatmap)
        # 高さを持たせるためにリピート
        activity_map = np.tile(all_spikes, (50, 1))
        
        ax.imshow(activity_map, aspect='auto', cmap='inferno', vmin=0, vmax=1)
        ax.set_yticks([])
        
        # 領域の境界線とラベル
        for i, boundary in enumerate(boundaries[:-1]):
            # 境界線
            if i > 0:
                ax.axvline(x=boundary, color='white', linestyle='--', alpha=0.5)
            # ラベル (中央に配置)
            center = (boundaries[i] + boundaries[i+1]) / 2
            ax.text(center, 60, labels[i], color='black', ha='center', fontsize=9, fontweight='bold')

        ax.set_title("Neural Activity Snapshot (Real-time)")
        plt.tight_layout()
        
        # 画像バッファへの書き出し
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        
        # PIL -> Numpy
        image = Image.open(buf)
        return np.array(image)