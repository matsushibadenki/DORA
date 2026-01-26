# ファイルパス: scripts/experiments/brain/run_phase2_autonomous_agent.py
import os
import time
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Type, cast

from snn_research.cognitive_architecture.artificial_brain import ArtificialBrain
from snn_research.utils.config_loader import load_config

# Mypy対策: 条件付きインポートの型定義
try:
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    # 開発環境でomegaconfがない場合のフォールバック定義
    DictConfig = dict  # type: ignore
    OmegaConf = None   # type: ignore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Phase2Agent")

class Phase2AutonomousAgent:
    def __init__(self, config_path: str):
        # Configロード
        raw_config = load_config(config_path)
        
        # DictConfig -> Dict 変換
        self.config: Dict[str, Any] = {}
        
        if OmegaConf is not None and isinstance(raw_config, DictConfig):
            # OmegaConfが利用可能な場合
            container = OmegaConf.to_container(raw_config, resolve=True)
            if isinstance(container, dict):
                # Mypy修正: 明示的なキャスト
                self.config = cast(Dict[str, Any], container)
        elif isinstance(raw_config, dict):
            self.config = raw_config
        else:
            # 万が一の場合
            self.config = {}

        self.device = torch.device(str(self.config.get("device", "cpu")))
        
        # 脳の初期化
        self.brain = ArtificialBrain(self.config)
        self.brain.to(self.device)
        
        # 実験パラメータ
        self.steps = 0
        self.max_steps = 1000 
        
        # 安全なアクセス
        cognitive_conf = self.config.get('cognitive', {})
        if isinstance(cognitive_conf, dict):
            sleep_conf = cognitive_conf.get('sleep', {})
            if isinstance(sleep_conf, dict):
                self.sleep_interval = int(sleep_conf.get('cycle_interval', 100))
            else:
                self.sleep_interval = 100
        else:
            self.sleep_interval = 100
        
        # 型ヒント追加
        self.stats: Dict[str, List[float]] = {"stability": [], "energy": []}

    def run_life_cycle(self):
        logger.info("Starting Phase 2 Autonomous Life Cycle...")
        try:
            while self.steps < self.max_steps:
                self.steps += 1
                if self.steps % self.sleep_interval == 0:
                    self._run_sleep_cycle()
                else:
                    self._run_awake_cycle()
                
                if self.steps % 10 == 0:
                    self._report_status()
                    
        except KeyboardInterrupt:
            logger.info("Simulation stopped by user.")
        finally:
            self._save_results()

    def _get_sensory_input(self) -> torch.Tensor:
        # 安全なアクセス
        model_conf = self.config.get('model', {})
        input_size = 64
        if isinstance(model_conf, dict):
            net_conf = model_conf.get('network', {})
            if isinstance(net_conf, dict):
                sizes = net_conf.get('layer_sizes', [64])
                if isinstance(sizes, list) and len(sizes) > 0:
                    input_size = sizes[0]
        
        prob = 0.2
        input_data = (torch.rand(1, input_size) < prob).float().to(self.device)
        return input_data

    def _run_awake_cycle(self):
        if not self.brain.is_awake:
            self.brain.wake_up()
        
        sensory_input = self._get_sensory_input()
        output = self.brain(sensory_input) # forward呼び出し
        
        metrics = self.brain.get_metrics()
        sparsity = metrics.get("sparsity_loss", 0.0)
        self.stats["energy"].append(sparsity)
        
        if torch.isnan(output).any():
            logger.error("Stability Collapse Detected: NaN values in output!")
            
        active_neurons = (output > 0).float().sum().item()
        total_neurons = output.numel()
        firing_rate = active_neurons / (total_neurons + 1e-6)
        
        if firing_rate > 0.10: 
            logger.warning(f"High Activity Warning: {firing_rate:.2%} active")

    def _run_sleep_cycle(self):
        self.brain.sleep()
        
        sleep_duration = 0.1
        cognitive_conf = self.config.get('cognitive', {})
        if isinstance(cognitive_conf, dict):
            sleep_conf = cognitive_conf.get('sleep', {})
            if isinstance(sleep_conf, dict):
                sleep_duration = float(sleep_conf.get('min_sleep_duration', 0.1))
        
        logger.info(f"Sleeping for {sleep_duration} virtual seconds...")
        time.sleep(0.1) 
        self.brain.wake_up()

    def _report_status(self):
        if self.stats["energy"]:
            avg_energy = np.mean(self.stats["energy"][-10:])
            logger.info(f"Step {self.steps}: Avg Energy(SparsityLoss)={avg_energy:.4f}")

    def _save_results(self):
        save_path = Path("workspace/runs/phase2_results.yaml")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_energy = float(np.mean(self.stats["energy"])) if self.stats["energy"] else 0.0
        
        with open(save_path, 'w') as f:
            yaml.dump({
                "steps_completed": self.steps,
                "final_energy_metric": final_energy,
                "status": "Completed"
            }, f)
        logger.info(f"Results saved to {save_path}")

def main():
    config_path = "configs/experiments/brain_v14_config.yaml"
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    agent = Phase2AutonomousAgent(config_path)
    agent.run_life_cycle()

if __name__ == "__main__":
    main()