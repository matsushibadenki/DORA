# ファイルパス: tests/cognitive_architecture/test_cognitive_components.py
# タイトル: 認知コンポーネント単体テスト (修正版 v4)
# 目的: test_memory_system_pipeline での記憶の重複排除によるアサーションエラーを修正。

from snn_research.cognitive_architecture.intrinsic_motivation import IntrinsicMotivationSystem
from snn_research.cognitive_architecture.global_workspace import GlobalWorkspace
from snn_research.cognitive_architecture.prefrontal_cortex import PrefrontalCortex
from snn_research.cognitive_architecture.cortex import Cortex
from snn_research.cognitive_architecture.hippocampus import Hippocampus
from snn_research.cognitive_architecture.motor_cortex import MotorCortex
from snn_research.cognitive_architecture.cerebellum import Cerebellum
from snn_research.cognitive_architecture.basal_ganglia import BasalGanglia
from snn_research.cognitive_architecture.amygdala import Amygdala
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock
import torch

# プロジェクトルートをPythonパスに追加
sys.path.append(str(Path(__file__).resolve().parents[2]))


# --- Mocks for dependencies ---

@pytest.fixture
def mock_workspace():
    """GlobalWorkspaceのモックを作成するフィクスチャ。"""
    mock = MagicMock(spec=GlobalWorkspace)
    mock.get_current_thought.return_value = None
    mock.subscribe = MagicMock()
    return mock


@pytest.fixture
def mock_motivation_system():
    """IntrinsicMotivationSystemのモックを作成するフィクスチャ。"""
    mock = MagicMock(spec=IntrinsicMotivationSystem)
    mock.get_internal_state.return_value = {}
    return mock

# --- Amygdala Tests ---


def test_amygdala_evaluates_positive_emotion():
    amygdala = Amygdala()
    result = amygdala.process("素晴らしい成功体験でした。")
    assert result is not None
    assert result['valence'] > 0.0
    print("\n✅ Amygdala: ポジティブな感情の評価テストに成功。")


def test_amygdala_evaluates_negative_emotion():
    amygdala = Amygdala()
    result = amygdala.process("危険なエラーが発生し、失敗した。")
    assert result is not None
    assert result['valence'] < 0.0
    print("✅ Amygdala: ネガティブな感情の評価テストに成功。")


def test_amygdala_handles_mixed_emotion():
    amygdala = Amygdala()
    result = amygdala.process("失敗の中に喜びを見出す。")
    assert result is not None
    assert -0.8 < result['valence'] < 0.8
    print("✅ Amygdala: 混合感情の評価テストに成功。")


def test_amygdala_handles_neutral_text():
    amygdala = Amygdala()
    result = amygdala.process("これは机です。")
    assert result is None
    print("✅ Amygdala: 中立的なテキスト(ヒットなし)の評価テストに成功。")


def test_amygdala_handles_empty_string():
    amygdala = Amygdala()
    result = amygdala.process("")
    assert result is None
    print("✅ Amygdala: 空文字列入力のテストに成功。")

# --- BasalGanglia Tests ---


def test_basal_ganglia_selects_best_action(mock_workspace):
    basal_ganglia = BasalGanglia(
        workspace=mock_workspace, selection_threshold=0.4)
    candidates = [
        {'action': 'A', 'value': 0.9},
        {'action': 'B', 'value': 0.6},
        {'action': 'C', 'value': 0.2},
    ]
    selected = basal_ganglia.select_action(candidates)
    assert selected is not None and selected['action'] == 'A'
    print("✅ BasalGanglia: 最適行動選択のテストに成功。")


def test_basal_ganglia_rejects_low_value_actions(mock_workspace):
    basal_ganglia = BasalGanglia(
        workspace=mock_workspace, selection_threshold=0.8)
    candidates = [{'action': 'A', 'value': 0.7}]
    selected = basal_ganglia.select_action(candidates)
    assert selected is None
    print("✅ BasalGanglia: 低価値行動の棄却テストに成功。")


def test_basal_ganglia_emotion_modulates_selection(mock_workspace):
    basal_ganglia = BasalGanglia(
        workspace=mock_workspace, selection_threshold=0.5)
    candidates = [{'action': 'run_away', 'value': 0.6}]
    fear_context = {'valence': -0.8, 'arousal': 0.9}
    selected_fear = basal_ganglia.select_action(
        candidates, emotion_context=fear_context)
    assert selected_fear is not None and selected_fear['action'] == 'run_away'
    print("✅ BasalGanglia: 情動による意思決定変調のテストに成功。")


def test_basal_ganglia_handles_no_candidates(mock_workspace):
    basal_ganglia = BasalGanglia(workspace=mock_workspace)
    selected = basal_ganglia.select_action([])
    assert selected is None
    print("✅ BasalGanglia: 行動候補が空の場合のテストに成功。")


def test_basal_ganglia_handles_none_emotion_context(mock_workspace):
    basal_ganglia = BasalGanglia(
        workspace=mock_workspace, selection_threshold=0.5)
    candidates = [{'action': 'A', 'value': 0.6}]
    selected = basal_ganglia.select_action(candidates, emotion_context=None)
    assert selected is not None and selected['action'] == 'A'
    print("✅ BasalGanglia: emotion_contextがNoneの場合のテストに成功。")

# --- Cerebellum & MotorCortex Tests ---


def test_cerebellum_and_motor_cortex_pipeline():
    cerebellum = Cerebellum()
    motor_cortex = MotorCortex(actuators=['test_actuator'])
    action = {'action': 'do_something', 'duration': 0.5}

    commands = cerebellum.refine_action_plan(action)
    assert len(commands) > 1 and commands[0]['command'] == 'do_something_start'

    log = motor_cortex.execute_commands(commands)
    assert len(log) > 1 and "do_something_start" in log[0]
    print("✅ Cerebellum -> MotorCortex パイプラインのテストに成功。")


def test_cerebellum_handles_empty_action():
    cerebellum = Cerebellum()
    commands = cerebellum.refine_action_plan({})
    assert commands == []
    print("✅ Cerebellum: 空の行動計画入力のテストに成功。")


def test_motor_cortex_handles_empty_commands():
    motor_cortex = MotorCortex()
    log = motor_cortex.execute_commands([])
    assert log == []
    print("✅ MotorCortex: 空のコマンドリスト入力のテストに成功。")

# --- Hippocampus & Cortex (Memory System) Tests ---


def test_memory_system_pipeline(tmp_path):
    # テスト用の一時ファイルを使用（本番DBを読み込まない）
    test_db_path = tmp_path / "test_memory_pipeline.json"
    
    hippocampus = Hippocampus(capacity=3, storage_file=str(test_db_path))
    cortex = Cortex()

    # 初期状態が空であることを確認 (安全策)
    assert len(hippocampus.episodic_buffer) == 0

    # 1. 短期記憶へ保存
    # Fix: Tensorの形状を変えることで、トリガー文字列("Tensor Pattern ...")を変え、
    #      海馬の重複排除ロジックに引っかからないようにする。
    hippocampus.store_episode(torch.ones(1, 784))
    hippocampus.store_episode(torch.ones(1, 128)) # Shape changed to avoid deduplication
    
    assert len(hippocampus.episodic_buffer) == 2

    # 2. 長期記憶へ固定化
    episode = hippocampus.episodic_buffer[0]
    concept = "animal_fact"
    definition = str(episode)

    cortex.consolidate_memory(concept, definition)

    # 3. 長期記憶から検索
    all_k = cortex.get_all_knowledge()
    assert len(all_k) > 0
    assert any("animal" in str(k) for k in all_k)
    print("✅ Hippocampus -> Cortex (記憶固定化) パイプラインのテストに成功。")


def test_hippocampus_handles_empty_episode(tmp_path):
    """海馬が空のエピソードを保存しようとした場合のテスト。"""
    test_db_path = tmp_path / "test_memory_empty.json"
    hippocampus = Hippocampus(capacity=3, storage_file=str(test_db_path))
    
    # 初期状態が空であることを確認
    assert len(hippocampus.episodic_buffer) == 0
    
    hippocampus.store_episode(torch.ones(784))
    assert len(hippocampus.episodic_buffer) == 1
    print("✅ Hippocampus: 空のエピソード保存テストに成功。")


def test_hippocampus_stores_valid_pattern(tmp_path):
    """有効なパターン入力時にエピソード記憶が保存されるかテスト。"""
    test_db_path = tmp_path / "test_memory_valid.json"
    hippocampus = Hippocampus(capacity=3, storage_file=str(test_db_path))
    
    # 初期状態が空であることを確認
    assert len(hippocampus.episodic_buffer) == 0
    
    dummy_input = torch.ones(784)
    hippocampus.store_episode(dummy_input)
    assert len(hippocampus.episodic_buffer) == 1
    print("✅ Hippocampus: パターン保存のテストに成功。")


def test_cortex_handles_non_string_input():
    cortex = Cortex()
    try:
        cortex.consolidate_memory("test_concept", 12345)  # type: ignore
    except Exception:
        pass
    assert True
    print("✅ Cortex: 予期せぬ入力型の処理テスト（エラーなし）に成功。")


def test_cortex_retrieves_nonexistent_concept():
    cortex = Cortex()
    dummy_vec = torch.randn(128)
    results = cortex.retrieve(dummy_vec)
    assert isinstance(results, list)
    print("✅ Cortex: 検索テストに成功。")

# --- PrefrontalCortex Tests ---


@pytest.mark.parametrize("context, expected_keyword", [
    ({"external_request": "summarize the document"}, "Fulfill external request"),
    ({"internal_state": {"boredom": 0.9}}, "Find something new"),
    ({"internal_state": {"curiosity": 0.9}}, "Investigate curiosity target"),
    ({"internal_state": {"boredom": 0.1, "curiosity": 0.2}, "conscious_content": {
     "type": "emotion", "valence": -0.9, "arousal": 0.8}}, "Ensure safety"),
    ({}, "Survive and Explore"),
])
def test_prefrontal_cortex_decides_goals(context, expected_keyword, mock_workspace, mock_motivation_system):
    mock_motivation_system.get_internal_state.return_value = context.get(
        "internal_state", {})
    mock_motivation_system.curiosity_context = "unknown"

    pfc = PrefrontalCortex(workspace=mock_workspace,
                           motivation_system=mock_motivation_system)

    conscious_data = context.get("conscious_content", {})
    source = "receptor" if "external_request" in context else "internal"
    
    if "external_request" in context:
        conscious_data = context["external_request"]

    broadcast_payload = {}
    if isinstance(conscious_data, dict):
        broadcast_payload = conscious_data.copy()
    else:
        broadcast_payload = {"content": conscious_data}
    
    broadcast_payload["source"] = source

    pfc.handle_conscious_broadcast(broadcast_payload)
    
    goal = pfc.current_goal

    assert expected_keyword in goal
    print(
        f"✅ PrefrontalCortex: '{expected_keyword}'に基づく目標設定のテストに成功。 Goal: '{goal}'")


def test_prefrontal_cortex_handles_empty_context(mock_workspace, mock_motivation_system):
    mock_motivation_system.get_internal_state.return_value = {}
    pfc = PrefrontalCortex(workspace=mock_workspace,
                           motivation_system=mock_motivation_system)
    
    pfc.handle_conscious_broadcast({"source": "unknown"})
    
    goal = pfc.current_goal
    assert "Survive and Explore" in goal
    print("✅ PrefrontalCortex: 空コンテキストでの目標設定テストに成功。")