# directory: snn_research/systems/sara_system.py
# title: SARA System Wrapper (Fix Import)
# description: インポートパスを相対パスから絶対パスに変更し、ModuleNotFoundErrorを回避した修正版。

import os
import sys
import numpy as np

# パス解決のための安全策：このファイルのある場所から見てルート(DORA)をsys.pathに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

# 絶対インポートに変更
from snn_research.models.sara_engine import SaraEngine

class SaraSystem:
    def __init__(self, input_size=784, output_size=10, model_path=None):
        self.engine = SaraEngine(input_size, output_size, load_path=model_path)
        
    def train_sample(self, img_flat: np.ndarray, label: int, time_steps=50):
        """画像1枚を学習"""
        spike_train = self._to_poisson(img_flat, time_steps)
        self.engine.train_step(spike_train, label, dropout_rate=0.1)
        
    def predict_sample(self, img_flat: np.ndarray, time_steps=50) -> int:
        """画像1枚を推論"""
        spike_train = self._to_poisson(img_flat, time_steps)
        return self.engine.predict(spike_train)
        
    def sleep(self):
        """睡眠フェーズ実行"""
        self.engine.sleep_phase()
        
    def save(self, filepath: str):
        """モデル保存"""
        self.engine.save_model(filepath)
        
    def _to_poisson(self, img_flat, time_steps):
        """内部ヘルパー: ポアソン変換"""
        img_flat = np.maximum(0, img_flat)
        max_val = np.max(img_flat)
        if max_val > 0: img_flat /= max_val
        
        rate = img_flat * 0.4
        spike_train = []
        for _ in range(time_steps):
            fired = np.where(np.random.rand(len(img_flat)) < rate)[0].tolist()
            spike_train.append(fired)
        return spike_train