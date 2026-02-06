# **DORA Development Roadmap 🗺️**

"No Matrix, No BP, No GPU" を実現したDORAカーネルを基盤とし、真の「Neuromorphic OS」へと進化させるためのロードマップ。

## **✅ Phase 1: Proof of Concept (Completed)**

**目標:** 行列演算を使わないイベント駆動SNNの実証。

* \[x\] **Event-Driven Simulator:** Pythonによる DORAKernel の実装。行列積を廃止し、隣接リストとイベントキューによる計算を実現。  
* \[x\] **Bio-Plausible Learning:** 誤差逆伝播（BP）を廃止し、予測誤差とSTDPによる局所学習を実装。  
* \[x\] **Stability:** 不応期と抑制性ニューロン（Dale's Law）導入による活動安定化。  
* \[x\] **Visualization:** ラスタープロットによる脳活動の可視化。  
* \[x\] **OS Integration:** 同期/非同期実行モードを備えた NeuromorphicOS の実装。

## **🚀 Phase 2: The Rust Engine (Current Focus)**

**目標:** Pythonの速度的限界を突破し、大規模ネットワーク（10万ニューロン〜）をリアルタイム動作させる。

* \[ \] **Rust Kernel:** カーネルコアロジック（event\_driven\_simulator.py）をRust言語で書き換え。  
* \[ \] **PyO3 Binding:** PythonからRustカーネルを透過的に呼び出せるラッパーの作成。  
* \[ \] **Memory Optimization:** メモリアリーナを用いたキャッシュ効率の良いノード管理。  
* \[ \] **SIMD Acceleration:** イベント処理のバッチ化とベクトル命令による高速化。

## **🌱 Phase 3: Structural Plasticity (The Living Brain)**

**目標:** 固定されたネットワークトポロジーからの脱却。学習＝「重みの変更」から「回路の組み替え」へ。

* \[ \] **Synaptogenesis:** 共発火するニューロン間に、物理的に新しいシナプスを動的に生成する機能。  
* \[ \] **Pruning:** 長期間使用されていない、あるいは強度が低いシナプスを物理的に削除し、メモリを解放する機能。  
* \[ \] **Dynamic Routing:** タスクに応じて情報の流れる経路が物理的に変化するアーキテクチャ。

## **🤖 Phase 4: Embodied & Social Intelligence**

**目標:** 身体性と社会性を持つエージェントへの進化。

* \[ \] **DVS Support:** イベントカメラ（Dynamic Vision Sensor）からの非同期ストリーム入力を直接処理するインターフェース。  
* \[ \] **Motor Control:** スパイク頻度によるサーボモータ等のアクチュエータ制御。  
* \[ \] **Multi-Agent Simulation:** 複数のDORAインスタンスが、言語（または独自の信号）を通じて知識を交換する「社会」のシミュレーション。

## **🌌 Phase 5: The Omega Point**

**目標:** 創発的意識の探求。

* \[ \] **Global Workspace Theory:** 大規模な再帰結合ネットワークにおける、情報統合と「意識」の創発現象の観測。  
* \[ \] **Self-Reflection:** 自己の内部状態（発火パターン）を言語化し、それを再び入力としてフィードバックするループの実装。

*Roadmap updated: 2026-02-06*