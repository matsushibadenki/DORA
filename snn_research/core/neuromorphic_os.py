# ファイルパス: snn_research/core/neuromorphic_os.py
# 日本語タイトル: Neuromorphic Research OS Kernel v3.2 (Fix AttributeError)
# 目的・内容:
#   Neuromorphic OSの最上位コンテナ。
#   v3.2修正: 旧コード互換性のため、deviceプロパティを追加し、
#   self.hardware.device への委譲を行うように修正。

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Core Layers
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.neuromorphic_scheduler import NeuromorphicScheduler
from snn_research.core.snn_core import SpikingNeuralSubstrate

logger = logging.getLogger(__name__)


class HardwareAbstractionLayer:
    """
    ハードウェア依存部分を吸収するレイヤ。
    CPU/GPU/MPS(Mac) などを自動判定し、デバイスオブジェクトを提供する。
    将来的にLoihi等の専用チップへのインターフェースもここに実装する。
    """

    def __init__(self, request_device: Optional[str]):
        self.device = self._select_device(request_device)
        self.device_name = str(self.device)

    def _select_device(self, device_name: Union[str, None]) -> torch.device:
        """デバイス選択ロジック"""
        if not device_name or device_name == "auto" or str(device_name).lower() == "none":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        try:
            return torch.device(device_name)
        except Exception as e:
            logger.warning(f"Device selection failed: {e}. Fallback to CPU.")
            return torch.device("cpu")


class NeuromorphicOS(nn.Module):
    """
    Neuromorphic Research OS (NROS) v3.2
    
    知能現象を「実装」するのではなく、神経ダイナミクスとして「観測」するための統合基盤。
    """

    def __init__(self, config: Dict[str, Any], device_name: Optional[str] = "auto"):
        super().__init__()
        self.config = config or {}

        # --- Layer 1: Hardware Abstraction ---
        self.hardware = HardwareAbstractionLayer(device_name)
        logger.info(f"🖥️ Neuromorphic OS booting on hardware: {self.hardware.device_name}")

        # --- Layer 2: Spiking Neural Substrate (The Kernel) ---
        # 全ての学習則とニューロン演算はここで行われる
        self.substrate = SpikingNeuralSubstrate(self.config, self.hardware.device)
        self._build_default_substrate()

        # --- Layer 3: Cognitive Architecture ---
        # 意識の放送、リソース管理などを行うOSサービス群
        self.global_workspace = GlobalWorkspace(dim=self.config.get("dim", 64))
        
        # Schedulerにはエネルギー管理用のアストロサイト機能（仮）として自分自身を渡す設計も可能だが、
        # ここでは簡易的にNoneまたはダミーを渡す構造にしておく
        class DummyAstrocyte:
            current_energy = 100.0

            def request_resource(self, name: str, amount: float) -> bool:
                return True

            def get_diagnosis_report(self) -> Dict[str, Any]:
                return {"metrics": {"inhibition_level": 0.0}}

        self.astrocyte = DummyAstrocyte()
        self.scheduler = NeuromorphicScheduler(self.astrocyte, self.global_workspace)

        # --- Layer 4: Observation & Logging ---
        self.system_status = "BOOTING"
        self.cycle_count = 0
        self.logs: List[Dict[str, Any]] = []

    @property
    def device(self) -> torch.device:
        """
        互換性のためのプロパティ。
        app/deployment.py 等が self.brain.device を参照する場合に対応。
        """
        return self.hardware.device

    def _build_default_substrate(self) -> None:
        """
        デフォルトの神経回路網構成（V1 -> Association -> Motor）。
        実験に応じて構成ファイルから読み込むべきだが、ここでは最小構成を定義。
        """
        input_dim = self.config.get("input_dim", 784)
        hidden_dim = self.config.get("hidden_dim", 256)
        output_dim = self.config.get("output_dim", 10)

        # 領域の作成
        self.substrate.add_neuron_group("V1", input_dim)
        self.substrate.add_neuron_group("Association", hidden_dim)
        self.substrate.add_neuron_group("Motor", output_dim)

        # 投射の作成 (PlasticityRuleは外部から注入可能)
        # 循環参照を避けるためメソッド内でインポート
        from snn_research.learning_rules.forward_forward import ForwardForwardRule
        
        ff_rule = ForwardForwardRule(learning_rate=0.01)

        self.substrate.add_projection(
            "v1_to_assoc", "V1", "Association", plasticity_rule=ff_rule
        )
        self.substrate.add_projection(
            "assoc_to_motor", "Association", "Motor", plasticity_rule=ff_rule
        )

        logger.info("🧠 Default neural substrate topology built.")

    def boot(self) -> None:
        """システム起動シーケンス"""
        self.substrate.reset_state()
        self.system_status = "RUNNING"
        self.cycle_count = 0
        logger.info("🚀 Neuromorphic OS Kernel started. Ready for experiments.")

    def run_cycle(self, sensory_input: torch.Tensor, phase: str = "wake") -> Dict[str, Any]:
        """
        OSのメインループ（1タイムステップ）。

        Args:
            sensory_input: 外部からの感覚入力 (Tensor)
            phase: 'wake' (学習・推論) or 'sleep' (整理・統合)
        """
        self.cycle_count += 1

        # 1. Input Processing
        inputs = {"V1": sensory_input.to(self.hardware.device)}

        # 2. Substrate Step (Dynamics & Plasticity)
        # 学習則への phase 伝達は kwargs 経由で行う
        substrate_state = self.substrate.forward_step(inputs, phase=phase)

        # 3. Cognitive Services
        # 活性化した情報をGlobal Workspaceへアップロード（簡易実装）
        # ここではAssociation野のスパイク活動を意識の候補とする
        assoc_spikes = substrate_state["spikes"].get("Association")
        if assoc_spikes is not None:
            # スパイクレートなどを情報としてアップロード
            self.global_workspace.upload_to_workspace(
                "Association", {"features": assoc_spikes}, salience=0.8
            )

        # 4. Scheduling (Optional)
        # 複雑なタスクがあればスケジューラを回す
        # scheduler_results = self.scheduler.step()

        # 5. Observation
        observation = {
            "cycle": self.cycle_count,
            "status": self.system_status,
            "phase": phase,
            "substrate_activity": {
                k: v.mean().item() for k, v in substrate_state["spikes"].items() if v is not None
            },
            "consciousness": self.global_workspace.get_current_thought().mean().item(),
        }

        return observation

    def shutdown(self) -> None:
        """システム終了処理"""
        self.system_status = "SHUTDOWN"
        logger.info("💤 Neuromorphic OS shutting down.")