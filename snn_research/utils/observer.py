# ファイルパス: snn_research/utils/observer.py
# 日本語タイトル: Experiment Observer & Logger
# 目的: 実験中のメトリクス収集、ログ保存、簡易可視化を一元管理する

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime


class ExperimentObserver:
    """
    実験のメトリクスを収集・保存・可視化するクラス。

    Attributes:
        save_dir (str): ログ保存ディレクトリ
        experiment_name (str): 実験名（タイムスタンプ付きでディレクトリ生成に使用）
        metrics (Dict[str, List[Dict]]): 収集したメトリクスデータ
        logger (logging.Logger): ロガー
    """

    def __init__(self, base_dir: str = "benchmarks/results", experiment_name: str = "experiment"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{self.timestamp}"
        self.save_dir = os.path.join(base_dir, self.experiment_id)

        # Create directory
        os.makedirs(self.save_dir, exist_ok=True)

        # Setup Logging
        self.logger = logging.getLogger(self.experiment_id)
        self.logger.setLevel(logging.INFO)
        # File handler
        fh = logging.FileHandler(os.path.join(self.save_dir, "experiment.log"))
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(fh)
        # Stream handler (console) - avoid adding duplicate if root logger already has one, but ensures explicit control
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(sh)

        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.config: Dict[str, Any] = {}

        self.log(f"🟢 Observer initialized. ID: {self.experiment_id}")
        self.log(f"📂 Results will be saved to: {self.save_dir}")

    def log(self, message: str, level: str = "info"):
        """コンソールとファイルにログ出力"""
        if level.lower() == "info":
            self.logger.info(message)
        elif level.lower() == "warning":
            self.logger.warning(message)
        elif level.lower() == "error":
            self.logger.error(message)

    def set_config(self, config: Dict[str, Any]):
        """実験設定を保存"""
        self.config = config
        with open(os.path.join(self.save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4, default=str)

    def log_metric(self, name: str, value: float, step: int, phase: str = "train"):
        """
        メトリクスを記録する
        Args:
            name: メトリクス名 (e.g., "loss", "accuracy", "v1_goodness")
            value: 値
            step: ステップ数 or エポック数
            phase: "train", "val", "test" など
        """
        if name not in self.metrics:
            self.metrics[name] = []

        record = {
            "step": step,
            "value": float(value),  # Ensure python float for JSON
            "phase": phase,
            "timestamp": time.time()
        }
        self.metrics[name].append(record)

    def save_results(self):
        """現在の全メトリクスをJSONファイルに保存"""
        filepath = os.path.join(self.save_dir, "metrics.json")
        try:
            with open(filepath, "w") as f:
                json.dump(self.metrics, f, indent=4)
            self.log(f"💾 Metrics saved to {filepath}")
        except Exception as e:
            self.log(f"⚠️ Failed to save metrics: {e}", "error")

    def plot_learning_curve(self, metric_names: Optional[List[str]] = None, title: str = "Learning Curve"):
        """
        記録されたメトリクスをプロットして保存する
        Args:
            metric_names: プロットしたいメトリクス名のリスト。Noneなら全て。
        """
        if not self.metrics:
            self.log("⚠️ No metrics to plot.")
            return

        target_metrics = metric_names if metric_names else list(
            self.metrics.keys())

        plt.figure(figsize=(10, 6))

        for name in target_metrics:
            if name not in self.metrics:
                continue

            data = self.metrics[name]
            steps = [d["step"] for d in data]
            values = [d["value"] for d in data]

            plt.plot(steps, values, label=name, marker='o', markersize=3)

        plt.title(f"{title} - {self.experiment_id}")
        plt.xlabel("Step/Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        plot_path = os.path.join(self.save_dir, "learning_curve.png")
        plt.savefig(plot_path)
        plt.close()
        self.log(f"📈 Plot saved to {plot_path}")

    def summary(self) -> str:
        """簡単なサマリー文字列を返す"""
        lines = ["--- Experiment Summary ---"]
        for name, data in self.metrics.items():
            if data:
                last_val = data[-1]["value"]
                max_val = max(d["value"] for d in data)
                lines.append(f"{name}: Last={last_val:.4f}, Max={max_val:.4f}")
        return "\n".join(lines)
