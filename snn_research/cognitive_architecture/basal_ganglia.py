# ファイルパス: snn_research/cognitive_architecture/basal_ganglia.py
# 日本語タイトル: Basal Ganglia Action Selector v2.2 (MyPy Fix)
# 目的: GlobalWorkspace.subscribe のコールバックシグネチャ不整合を修正。

from typing import List, Dict, Any, Optional
import torch
from .global_workspace import GlobalWorkspace


class BasalGanglia:
    workspace: GlobalWorkspace

    def __init__(self, workspace: GlobalWorkspace, selection_threshold: float = 0.5, inhibition_strength: float = 0.3):
        self.workspace = workspace
        self.base_threshold = selection_threshold
        self.inhibition_strength = inhibition_strength
        self.selected_action: Optional[Dict[str, Any]] = None

        self.workspace.subscribe(self.handle_conscious_broadcast)
        print("🧠 大脳基底核モジュールが初期化され、Workspaceを購読しました。")

    def handle_conscious_broadcast(self, broadcast_data: Dict[str, Any]) -> None:
        """
        MyPy Fix: 引数を辞書1つに変更。
        """
        # source = broadcast_data.get("source", "unknown")
        # 簡易実装としてログ出力のみ行う（現在はpass）
        pass

    def _generate_internal_candidates(self) -> List[Dict[str, Any]]:
        return [
            {'action': 'investigate_perception', 'value': 0.3},
            {'action': 'reflect_on_emotion', 'value': 0.2},
            {'action': 'ignore', 'value': 0.1},
        ]

    def _modulate_threshold(self, emotion_context: Optional[Dict[str, float]]) -> float:
        if emotion_context is None:
            return self.base_threshold

        arousal = emotion_context.get("arousal", 0.0)
        return max(0.1, min(0.9, self.base_threshold - arousal * 0.2))

    def select_action(
        self,
        external_candidates: List[Dict[str, Any]],
        emotion_context: Optional[Dict[str, float]] = None
    ) -> Optional[Dict[str, Any]]:
        self.selected_action = None

        internal_candidates = self._generate_internal_candidates()
        all_candidates = external_candidates + internal_candidates

        if not all_candidates:
            return None

        current_threshold = self._modulate_threshold(emotion_context)

        values = torch.tensor([float(c.get('value', 0.0)) for c in all_candidates])

        best_idx = int(torch.argmax(values).item())
        best_val = values[best_idx].item()

        if best_val >= current_threshold:
            self.selected_action = all_candidates[best_idx]
            return self.selected_action
        else:
            return None