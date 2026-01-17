# Neuromorphic Research OS: 実験・テストコマンドガイド (v6.1)

本ドキュメントでは、**Neuromorphic Research OS (NROS)** のすべての機能を検証・テストするためのコマンドを網羅しています。

開発者、リサーチャーは、変更を加えた後にこれらのテストを実行し、システムの健全性と性能を確認してください。

## ⚠️ 実行環境について

*   **ルートディレクトリ**: すべてのコマンドはプロジェクトルート (`DORA/`) で実行してください。
*   **ファイル出力**: テスト結果やログは主に `workspace/` ディレクトリに出力されます。

## 1. システム健全性・基本機能テスト (System Health)

まず最初に実行すべき、システムの基本的な動作確認コマンドです。

### プロジェクト健全性チェック
必須ファイル、ディレクトリ構造、依存ライブラリの確認を行います。

```bash
python scripts/tests/run_project_health_check.py
```

### 全ユニットテスト実行 (Pytest)
各モジュールの単体テストを一括実行します。

```bash
# Pytestを使用 (推奨)
python -m pytest tests/

# または専用ランナーを使用
python scripts/tests/run_all_tests.py
```

### コンパイラ・ハードウェア抽象化層テスト
SNNモデルのコンパイルとデバイス割り当てのテストです。

```bash
python scripts/tests/run_compiler_test.py
```

## 2. 性能・安定性ベンチマーク (Benchmarks)

学習の安定性とシステムの処理速度を定量的に評価します。

### 学習安定性ベンチマーク (Stability Benchmark)
SNNが破綻せずに学習できるかを検証します。リファクタリング後の必須テストです。

```bash
# 標準設定 (5回の試行)
python benchmarks/stability_benchmark_v2.py --runs 5 --epochs 5 --threshold 90
```
*   **出力**: `workspace/benchmarks/stability_benchmark_results.json`

### レイテンシ測定
推論の応答速度を測定します。

```bash
python scripts/benchmarks/benchmark_latency.py
```

### ベンチマークスイート
複数のベンチマークをまとめて実行します。

```bash
python scripts/benchmarks/run_benchmark_suite.py
```

## 3. 統合実験・機能検証 (Experiments & Features)

OSの主要機能（覚醒・睡眠・可塑性など）が正しく機能しているかを確認する実験スクリプトです。

### ライフサイクル実験 (Wake/Sleep Cycle)
自律的な覚醒と睡眠のサイクル、およびエネルギー代謝を検証します。

```bash
python scripts/experiments/run_research_cycle.py
```
*   **確認事項**: ログ (`workspace/runtime_state/`) でエネルギー減少と睡眠移行を確認。

### 記憶定着実験 (Memory Consolidation)
学習→睡眠（リプレイ）→再学習による記憶転送とシナプス可塑性を検証します。

```bash
python scripts/experiments/learning/run_memory_consolidation.py
```

### 社会性・集合知 (Social Tests)
複数のエージェント間のインタラクションや言語創発のテストです。

*   **ネーミングゲーム (言語創発)**:
    ```bash
    python scripts/experiments/social/run_naming_game.py
    ```
*   **集合知 (Collective Intelligence)**:
    ```bash
    python scripts/experiments/systems/run_collective_intelligence.py
    ```

### システム拡張性 (Scalability)
大規模なネットワーク構成での動作検証です。

```bash
python scripts/tests/verify_scalability.py
```

## 4. デモンストレーション (Demos as Tests)

特定の脳機能モジュールが正しく動作することを目視で確認するためのデモです。

### 視覚・知覚
*   **Spiking Forward-Forward**: `scripts/demos/visual/run_spiking_ff_demo_v2.py`
    *   True SNNでの学習動作確認。
*   **空間認識**: `scripts/demos/visual/run_spatial_demo.py`

### 脳機能
*   **世界モデル (World Model)**: `scripts/demos/brain/run_world_model_demo.py`
*   **意識のブロードキャスト**: `scripts/demos/brain/run_conscious_broadcast_demo.py`
*   **自由意志**: `scripts/demos/brain/run_free_will_demo.py`

### 学習メカニズム
*   **継続学習**: `scripts/demos/learning/run_continual_learning_demo.py`
*   **睡眠学習**: `scripts/demos/learning/run_sleep_learning_demo.py`

## 5. デバッグ・診断ツール (Debug)

開発時に詳細な内部状態を確認するためのツールです。

*   **スパイク活動モニタ**:
    ```bash
    python scripts/debug/debug_spike_activity.py
    ```
*   **シグナル診断**:
    ```bash
    python scripts/debug/diagnose_signal.py
    ```

## 6. クイック検証フロー

変更を加えた際の手っ取り早い確認フロー:

1.  `python scripts/tests/run_project_health_check.py` (環境確認)
2.  `python -m pytest tests/` (論理エラー確認)
3.  `python benchmarks/stability_benchmark_v2.py --runs 1` (動作・学習確認)
