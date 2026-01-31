# ファイルパス: snn_research/cognitive_architecture/prefrontal_cortex.py
# 日本語タイトル: 前頭前野モジュール v2.3 (MyPy Fix)
# 目的: GlobalWorkspace.subscribe のコールバックシグネチャ不整合を修正。

from __future__ import annotations
import logging
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, TYPE_CHECKING

# 循環インポート防止のため、実行時はインポートせず型チェック時のみ有効化
if TYPE_CHECKING:
    from .global_workspace import GlobalWorkspace
    from .intrinsic_motivation import IntrinsicMotivationSystem

logger = logging.getLogger(__name__)


class PrefrontalCortex:
    """
    実行制御（Executive Control）を司る前頭前野モジュール。
    """
    # 型アノテーションに文字列を使用し、実行時の依存を排除
    workspace: 'GlobalWorkspace'

    def __init__(
        self,
        workspace: 'GlobalWorkspace',
        motivation_system: 'IntrinsicMotivationSystem',
        d_model: int = 256,   # 高次元ベクトルの次元数
        device: str = 'cpu'
    ):
        self.workspace = workspace
        self.motivation_system = motivation_system
        self.d_model = d_model
        self.device = device

        # --- 既存の状態管理 ---
        self.current_goal: str = "Survive and Explore"
        self.current_context: str = "neutral"
        self.goal_stability: float = 0.0
        self.last_update_reason: str = "initialization"

        # --- 直交化・多重化のための幾何学的状態 ---
        self.uncertainty_axis = torch.randn(d_model, device=device)
        self.uncertainty_axis = F.normalize(self.uncertainty_axis, p=2, dim=0)

        raw_goal = torch.randn(d_model, device=device)
        self.goal_vector = self._project_orthogonally(
            raw_goal, self.uncertainty_axis)

        self.current_uncertainty_level: float = 0.0

        # ワークスペースのブロードキャストを購読
        if hasattr(self.workspace, 'subscribe'):
            self.workspace.subscribe(self.handle_conscious_broadcast)

        logger.info(
            f"🧠 Prefrontal Cortex (PFC) initialized with Orthogonal Geometry (d={d_model}).")

    def _project_orthogonally(self, target_vec: torch.Tensor, reference_axis: torch.Tensor) -> torch.Tensor:
        """
        [幾何学演算] グラム・シュミットの直交化プロセス。
        """
        projection = torch.dot(target_vec, reference_axis) * reference_axis
        orthogonal_vec = target_vec - projection
        return F.normalize(orthogonal_vec, p=2, dim=0)

    def handle_conscious_broadcast(self, broadcast_data: Dict[str, Any]) -> None:
        """
        ワークスペースからのブロードキャストを受け取り、エグゼクティブ・コントロールを更新する。
        MyPy Fix: 引数を辞書1つに変更。
        """
        source = str(broadcast_data.get("source", "unknown"))
        
        # 自身が発信源の情報は無視
        if source == "prefrontal_cortex":
            return

        # 動機付けシステムから現在の内部状態を取得
        internal_state = self.motivation_system.get_internal_state()

        # コンテキスト情報の構築
        # contentとしてbroadcast_data全体を渡す（中身にfeatures等が含まれる）
        context = {
            "source": source,
            "content": broadcast_data,
            "boredom": internal_state.get("boredom", 0.0),
            "curiosity": internal_state.get("curiosity", 0.0),
            "confidence": internal_state.get("confidence", 0.5)
        }

        self._update_executive_control(context)

    def _update_executive_control(self, context: Dict[str, Any]) -> None:
        """
        知覚や感情に基づいて、現在のゴールや行動指針を決定する。
        """
        source = context["source"]
        content = context["content"]

        # 1. 不確実性の推定
        confidence = context.get("confidence", 0.5)
        self.current_uncertainty_level = 1.0 - float(confidence)

        uncertainty_state_vec = self.uncertainty_axis * self.current_uncertainty_level

        # 2. メタ認知制御：柔軟性（Flexibility）の計算
        flexibility_gate = 1.0 - \
            torch.sigmoid(torch.tensor(
                (self.current_uncertainty_level - 0.5) * 5.0)).item()

        new_goal_text: Optional[str] = None
        reason: Optional[str] = None
        salience = 0.5
        force_update = False

        # --- ルールベース決定ロジック ---

        # A. 外部要求
        # contentはDictなので、特定のキーを見るか、文字列表現を確認
        content_str = str(content)
        if source == "receptor" or "request" in content_str.lower():
            req_text = content_str
            new_goal_text = f"Fulfill external request: {req_text[:50]}"
            reason = "external_demand"
            salience = 0.9
            force_update = True

        # B. 感情（恐怖・危機）
        elif isinstance(content, dict) and content.get("type") == "emotion":
            valence = float(content.get("valence", 0.0))
            arousal = float(content.get("arousal", 0.0))
            if valence < -0.7 and arousal > 0.6:
                new_goal_text = "Ensure safety / Avoid negative stimulus"
                reason = "fear_response"
                salience = 1.0
                force_update = True

        # C. 内発的動機
        elif not new_goal_text:
            if float(context["boredom"]) > 0.8:
                new_goal_text = "Find something new / Explore random"
                reason = "high_boredom"
                salience = 0.7
            elif float(context["curiosity"]) > 0.8:
                topic = getattr(self.motivation_system,
                                'curiosity_context', "unknown")
                new_goal_text = f"Investigate curiosity target: {str(topic)[:30]}"
                reason = "high_curiosity"
                salience = 0.8

        # --- ゴール更新処理 ---

        if new_goal_text:
            if new_goal_text == self.current_goal:
                return

            if not force_update and flexibility_gate < 0.3:
                logger.info(
                    f"🛡️ PFC Stability Check: Goal update suppressed due to high uncertainty (Flexibility: {flexibility_gate:.2f})")
                return

            safe_reason: str = reason if reason is not None else "context_change"

            logger.info(
                f"🤔 PFC Re-evaluating Goal: '{self.current_goal}' -> '{new_goal_text}' ({safe_reason})")

            self.current_goal = new_goal_text
            self.last_update_reason = safe_reason

            # ゴールベクトルの更新（シミュレーション）
            proto_goal_vec = torch.randn(self.d_model, device=self.device)
            self.goal_vector = self._project_orthogonally(
                proto_goal_vec, self.uncertainty_axis)

            pfc_state_vector = self.goal_vector + uncertainty_state_vec

            # ワークスペースへ新しいゴールを提示
            if hasattr(self.workspace, 'upload_to_workspace'):
                self.workspace.upload_to_workspace(
                    source_name="prefrontal_cortex",
                    content={
                        "features": pfc_state_vector.unsqueeze(0),
                        "type": torch.tensor([1.0]),
                        "goal_text": new_goal_text
                    },
                    salience=salience
                )

    def plan(self, conscious_content: Any) -> Optional[Dict[str, Any]]:
        """
        現在のゴールと意識の内容に基づいて、ハイレベルな行動計画を生成する。
        """
        plan_data = {
            "goal": self.current_goal,
            "reason": self.last_update_reason,
            "target": None,
            "directive": "monitor",
            "priority": 0.5
        }

        if self.current_uncertainty_level > 0.8:
            plan_data["directive"] = "observe_carefully"
            plan_data["reason"] = "high_uncertainty"
            return plan_data

        if isinstance(conscious_content, dict):
            if "features" in conscious_content:
                plan_data["target"] = "visual_object"
                plan_data["directive"] = "inspect_visual"
                plan_data["priority"] = 0.8
            elif "surprise" in conscious_content:
                plan_data["target"] = "anomaly"
                plan_data["directive"] = "resolve_surprise"
                plan_data["priority"] = 0.9

        elif isinstance(conscious_content, str):
            plan_data["target"] = "verbal_content"
            plan_data["directive"] = "process_language"

        return plan_data

    def get_executive_context(self) -> Dict[str, Any]:
        return {
            "goal": self.current_goal,
            "context": self.current_context,
            "reason": self.last_update_reason,
            "stability": self.goal_stability,
            "uncertainty_level": self.current_uncertainty_level,
            "vector_orthogonality": self._check_orthogonality()
        }

    def _check_orthogonality(self) -> float:
        dot_prod = torch.dot(self.goal_vector, self.uncertainty_axis)
        return dot_prod.item()