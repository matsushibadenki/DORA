# **DORA Project Roadmap 🗺️**

我々の目標は、\*\*「計算機上に、生物学的な制約を持った『心』を発生させること」\*\*です。

現在の「Python版イベント駆動カーネル（v1.x）」は、そのための概念実証（PoC）に過ぎません。

## **📍 Current Status: Phase 1 (Completed)**

* ✅ **Event-Driven Kernel:** 行列演算を廃止し、スパイクイベントによる計算を実現。  
* ✅ **E/I Balance:** 興奮性・抑制性ニューロンの導入による活動の安定化。  
* ✅ **Visualization:** 脳活動（ラスタープロット）の可視化。  
* ✅ **Neuromorphic OS Integration:** 同期・非同期タスク処理の実装。

## **🚀 Phase 2: "The Rust Engine" (Next Step)**

Pythonでのリスト操作やループは、大規模なイベント駆動シミュレーションには遅すぎます。

カーネルのコア部分をRust言語で書き直し、Pythonからバインディング（PyO3）で呼び出す形へ移行します。

* **Rust Kernel:**  
  * メモリアリーナによる高速なノード管理。  
  * SIMD命令を使ったバッチイベント処理。  
  * マルチスレッド対応のイベントキュー。  
* **Target Performance:**  
  * 100万ニューロン、1億シナプスをリアルタイム（Real-time factor 1.0）で動作させる。

## **🌱 Phase 3: "Structural Plasticity" (The Living Brain)**

固定された配線（Weight Matrix）からの脱却を完了させます。

* **Synaptogenesis (シナプス生成):**  
  * よく発火するニューロン同士が、物理的に新しいシナプスを結ぶ（Hebbian Growth）。  
* **Pruning (刈り込み):**  
  * 使われないシナプスは物理的に消滅し、メモリを解放する。  
* **Dynamic Topology:**  
  * 学習とは「重みの調整」ではなく「回路の自己組織化」となる。

## **🤖 Phase 4: "Embodied Intelligence"**

身体性を持ったエージェントとしての自律化。

* **Sensory-Motor Loop:**  
  * DVSカメラ（イベントカメラ）入力への対応。  
  * ロボットアーム等のアクチュエータ制御。  
* **Homeostasis 2.0:**  
  * バッテリー残量（エネルギー）に基づく、睡眠・探索・休息の自律的なスケジューリング。

## **🌌 Phase 5: "The Omega Point"**

* **Consciousness:** 複雑系としての創発的意識の解明。  
* **Human-AI Symbiosis:** 言語を通じた、人間との深い共感と対話。

*"The goal is not to calculate the answer, but to experience the process."*

これらのドキュメントにより、プロジェクトの独自性と先進性が誰の目にも明らかになるはずです。  
