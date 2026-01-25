# ファイルパス: snn_research/utils/observer.py
# 日本語タイトル: Advanced Neuromorphic Observer (Layer 4 Debugging Tool)
# 目的: 
#   実験メトリクスの収集に加え、ニューロンの発火ヒートマップ、システム内部状態のスナップショット、
#   デバッグ用ダッシュボード向けの構造化データ出力を提供する。

import os
import json
import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

class NeuromorphicObserver:
    """
    SNN OSのための高度な観測・デバッグツール。
    Layer 4 (Observation) の役割を担い、内部状態を可視化・永続化する。
    """

    def __init__(self, base_dir: str = "benchmarks/results", experiment_name: str = "snn_experiment"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        self.save_dir = os.path.join(base_dir, self.experiment_id)
        
        # ディレクトリ構造の作成
        self.dirs = {
            "logs": os.path.join(self.save_dir, "logs"),
            "plots": os.path.join(self.save_dir, "plots"),
            "heatmaps": os.path.join(self.save_dir, "plots/heatmaps"),
            "snapshots": os.path.join(self.save_dir, "snapshots"),
            "dashboard": os.path.join(self.save_dir, "dashboard_data")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        # Logger Setup
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        fh = logging.FileHandler(os.path.join(self.dirs["logs"], "system.log"))
        fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        self.logger.addHandler(fh)
        
        # コンソール出力（重複防止）
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('👁️ [OBSERVER] %(message)s'))
            self.logger.addHandler(sh)

        # データストア
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.system_events: List[Dict[str, Any]] = []
        self.config: Dict[str, Any] = {}
        
        self.log(f"Observer initialized. ID: {self.experiment_id}")

    def log(self, message: str, level: str = "info"):
        """ログ出力"""
        if level == "info": self.logger.info(message)
        elif level == "warning": self.logger.warning(message)
        elif level == "error": self.logger.error(message)

    def set_config(self, config: Dict[str, Any]):
        """設定の保存"""
        self.config = config
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, default=str)

    # --- Metrics & Events ---

    def log_metric(self, name: str, value: float, step: int, phase: str = "train"):
        """時系列数値データの記録 (Loss, Accuracy, Energyなど)"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({
            "step": step,
            "value": float(value),
            "phase": phase,
            "timestamp": time.time()
        })

    def log_event(self, event_type: str, details: Dict[str, Any], step: int):
        """システムイベントの記録 (Phase Change, Task Executionなど)"""
        event = {
            "step": step,
            "type": event_type,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.system_events.append(event)
        
        # 重大なイベントは即時ログ出力
        if event_type in ["phase_change", "error", "critical_alert"]:
            self.log(f"Event [{event_type}]: {details}")

    # --- Advanced Visualization (Layer 4) ---

    def log_heatmap(self, 
                    data: Union[np.ndarray, Any], 
                    name: str, 
                    step: int, 
                    vmin: Optional[float] = None, 
                    vmax: Optional[float] = None,
                    cmap: str = "viridis"):
        """
        行列データ（重み、注意マップ、発火頻度など）をヒートマップとして保存する。
        """
        # Tensor/List変換
        if hasattr(data, 'detach'):
            data = data.detach().cpu().numpy()
        elif isinstance(data, list):
            data = np.array(data)
            
        if data.ndim != 2:
            # 2次元でない場合は適当に整形またはスキップ
            if data.ndim == 1:
                data = data.reshape(1, -1)
            else:
                return # 3次元以上は今のところ未対応（スライスが必要）

        plt.figure(figsize=(10, 8))
        sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, square=False)
        plt.title(f"{name} (Step {step})")
        
        filename = f"{name}_step_{step:06d}.png"
        save_path = os.path.join(self.dirs["heatmaps"], filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

    def snapshot_system_state(self, scheduler_status: Dict, brain_status: Dict, step: int):
        """
        OSとBrainの全状態をスナップショットとしてJSON保存する。
        デバッガで時系列再生するために使用可能。
        """
        snapshot = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "scheduler": scheduler_status,
            "brain": brain_status,
            # 最新のメトリクス値を含める
            "latest_metrics": {k: v[-1]["value"] for k, v in self.metrics.items() if v}
        }
        
        filename = f"state_step_{step:06d}.json"
        with open(os.path.join(self.dirs["snapshots"], filename), "w") as f:
            json.dump(snapshot, f, indent=2, default=str)

    # --- Reporting ---

    def save_results(self):
        """全データの保存"""
        # Metrics
        with open(os.path.join(self.save_dir, "metrics.json"), "w") as f:
            json.dump(self.metrics, f, indent=4)
        
        # Events
        with open(os.path.join(self.save_dir, "system_events.json"), "w") as f:
            json.dump(self.system_events, f, indent=4)
            
        self.log(f"💾 All data saved to {self.save_dir}")

    def plot_learning_curve(self, metric_names: Optional[List[str]] = None):
        """学習曲線のプロット"""
        if not self.metrics: return
        
        target_metrics = metric_names if metric_names else list(self.metrics.keys())
        
        # フィルタリング（値がスカラのもののみ）
        valid_metrics = [m for m in target_metrics if m in self.metrics and len(self.metrics[m]) > 0]
        
        if not valid_metrics: return

        plt.figure(figsize=(12, 6))
        for name in valid_metrics:
            data = self.metrics[name]
            steps = [d["step"] for d in data]
            values = [d["value"] for d in data]
            plt.plot(steps, values, label=name, alpha=0.8)

        plt.xlabel("Step")
        plt.ylabel("Value")
        plt.title(f"Experiment Metrics - {self.experiment_id}")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        plt.savefig(os.path.join(self.dirs["plots"], "learning_curve.png"))
        plt.close()

    def generate_dashboard_data(self):
        """
        フロントエンド（Web UI等）で可視化するための集約データを生成する。
        """
        dashboard_summary = {
            "experiment_id": self.experiment_id,
            "duration": "Running...", 
            "metrics_summary": {},
            "event_log_tail": self.system_events[-50:] if self.system_events else []
        }
        
        for name, data in self.metrics.items():
            if data:
                values = [d["value"] for d in data]
                dashboard_summary["metrics_summary"][name] = {
                    "last": values[-1],
                    "min": min(values),
                    "max": max(values),
                    "avg": sum(values) / len(values)
                }
        
        with open(os.path.join(self.dirs["dashboard"], "summary.json"), "w") as f:
            json.dump(dashboard_summary, f, indent=2)

    def summary(self) -> str:
        lines = [f"--- Observer Summary: {self.experiment_id} ---"]
        for name, data in self.metrics.items():
            if data:
                last = data[-1]["value"]
                lines.append(f"{name}: {last:.4f}")
        return "\n".join(lines)