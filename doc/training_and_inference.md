# Neuromorphic Research OS: 学習・観測ガイド (v6.1)

本プロジェクトにおける「学習（Training）」は、単なるパラメータ最適化ではなく、**「経験に伴う脳構造の自己組織化」**と定義されます。また、「推論（Inference）」は静的な計算ではなく、**「リアルタイムの神経活動現象」**として扱われます。

このドキュメントでは、Neuromorphic OS上での学習プロセスの回し方、観測方法、そして利用可能なデモやレシピについて解説します。

## 1. ファイル保存場所の変更 (v6.1～)

プロジェクト構成の整理に伴い、生成されるファイルは以下のディレクトリに分類されます。

*   **`data/`**: データセット (MNIST/CIFAR-10など)
*   **`models/`**: 学習済みモデルやチェックポイント
*   **`workspace/`**: 
    *   `runtime_state/`: システムの実行状態、ログ、プログレス
    *   `benchmarks/`: ベンチマーク結果
    *   `results/`: その他の実験結果

## 2. OS上での学習サイクル (Structural Learning)

Neuromorphic OSは自動化された学習サイクルを持ちます。

### 学習のメカニズム

1.  **覚醒 (Wake):**
    *   **Forward-Forward則 / STDP:** 局所的な可塑性による学習。
    *   **ドーパミン (Dopamine):** 報酬ベースの強化。
    *   **エピソード記録:** 海馬への一時保存。
2.  **睡眠 (Sleep):**
    *   **海馬リプレイ (Replay):** 記憶の長期定着（皮質への転送）。
    *   **シナプス恒常性 (Pruning):** 不要な接続の削除（忘却）。
    *   **シナプス生成 (Synaptogenesis):** 新たな接続の生成。

### 実行コマンド

**記憶定着と構造変化の実験（推奨）:**
```bash
python scripts/experiments/learning/run_memory_consolidation.py
```

### 学習安定性へのフォーカス (Stability Benchmark)

学習が破綻せず、安定して精度を維持できるかを検証するためのベンチマークです。

```bash
python benchmarks/stability_benchmark_v2.py --runs 5 --epochs 5 --threshold 90
```

*   **Metric:** 成功率、平均精度、安定性スコア。
*   **Progress:** `workspace/runtime_state/benchmark_progress.json`
*   **Results:** `workspace/benchmarks/stability_benchmark_results.json`

## 3. 観測と分析 (Observation)

学習の結果は、正解率だけでなく**脳の状態変化**として評価します。

### 主要な観測指標

*   **Synapse Count:** 接続総数の減少（刈り込み）と増加（生成）。
*   **Energy / Fatigue:** 脳のエネルギー代謝リズム。
*   **Consciousness Level:** 情報統合の強度。

### 可視化

```bash
python scripts/visualization/plot_memory_learning.py
```

## 4. 単体モデルの学習 (Component Training)

特定のタスクに特化したモデル（Spiking CNNなど）の学習レシピです。

### 高精度レシピ (Recipes)

`snn_research/recipes/` 以下のスクリプトを使用します。

*   **MNIST 学習:**
    ```bash
    python -c "from snn_research.recipes.mnist import run_mnist_training; run_mnist_training()"
    ```
    *   結果: `workspace/results/best_mnist_metrics.json`, `best_mnist_snn.pth`
*   **CIFAR-10 学習:**
    ```bash
    python -c "from snn_research.recipes.cifar10 import run_cifar10_training; run_cifar10_training()"
    ```

### 汎用トレーナー

```bash
PYTHONPATH=. python scripts/training/train.py --model_config configs/models/small.yaml
```

## 5. デモンストレーション (Demos)

様々な脳機能のデモスクリプトが `scripts/demos/` に用意されています。

### 視覚・知覚 (Visual)
*   **Spiking Forward-Forward Demo:**
    ```bash
    python scripts/demos/visual/run_spiking_ff_demo_v2.py
    ```
    *   True SNNでのForward-Forward学習デモ。モデルは `models/checkpoints/` に保存されます。

### 脳機能・認知 (Brain)
*   **World Model:** `scripts/demos/brain/run_world_model_demo.py`
*   **Conscious Broadcast:** `scripts/demos/brain/run_conscious_broadcast_demo.py`
*   **Curiosity:** `scripts/demos/brain/run_curiosity_demo.py`

### 学習・記憶 (Learning)
*   **Sleep Cycle:** `scripts/demos/learning/run_sleep_cycle_demo.py`
*   **Continual Learning:** `scripts/demos/learning/run_continual_learning_demo.py`

## 6. Webインターフェース (App Demo)

学習した脳の挙動をブラウザ上で確認できます。

### 起動コマンド

```bash
python app/main.py
```
*   アクセス: http://localhost:8000

### 統合デモ (Unified Perception)

視覚・言語・運動野の連携デモを行います。

```bash
python app/unified_perception_demo.py
```
