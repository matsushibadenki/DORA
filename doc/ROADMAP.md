# **DORA Development Roadmap 🗺️**

**"Compute Only What Matters."**

DORA (Distributed Online Reactive Architecture) は、既存のディープラーニング（ANN）が抱える「消費電力」と「計算コスト」の課題を、脳の動作原理（イベント駆動・スパース性）と最新の工学的手法（BitNet・Mamba）の融合によって解決します。

我々の「勝ち筋」は、**「学習はANNの安定性を、推論はSNNの効率性を」** 手に入れるハイブリッド戦略にあります。

## **✅ Phase 1: Proof of Concept (Completed)**

**目標:** 行列演算を使わないイベント駆動SNNの基本動作と思想の実証。

* \[x\] **Event-Driven Simulator:** Pythonによる DORAKernel の実装。行列積を廃止し、隣接リストとイベントキューによる計算を実現。  
* \[x\] **Bio-Plausible Learning:** 誤差逆伝播（BP）を補完する、予測誤差とSTDPによる局所学習の実験。  
* \[x\] **Stability:** 不応期と抑制性ニューロン（Dale's Law）導入による活動安定化。  
* \[x\] **OS Integration:** 同期/非同期実行モードを備えた NeuromorphicOS の実装。

## **🚀 Phase 2: Hybrid Architecture & Stability (The "BitSpike" Era)**

**現在の焦点。**

**目標:** 「勾配の不安定さ」と「コンテキスト長の短さ」というSNNの弱点を克服し、LLMに対抗しうる対話能力と学習安定性を獲得する。

### **🔑 Key Strategy: BitNet \+ SNN**

1.58bit量子化技術（BitNet）をSNNに適用し、「足し算のみ」で動作しつつ、Transformer並みの学習安定性を目指す。

* \[ \] **BitSpikeMamba Implementation:**  
  * BitNet b1.58 ( {-1, 0, 1} の重み) と SNN ( {0, 1} の活性化) を組み合わせた BitSpikeMamba のアーキテクチャを確立。  
  * **課題克服:** RNN/SNNの弱点である「長期記憶（Long Context）」を、Mamba（状態空間モデル）の機構を取り入れることで解決する。  
* \[ \] **Stable Training Pipeline:**  
  * **Teacher Forcing:** 言語モデル学習において、正解データを入力しながら次のトークンを予測させるパイプラインの整備。  
  * **Surrogate Gradient Optimization:** スパイクの微分不可能性を回避する代理勾配法の最適化。  
* \[ \] **Neuro-Symbolic Integration (System 1 & 2):**  
  * 直感的なSNN（System 1）と、論理的なシンボリック推論/LLM（System 2）の連携強化。  
  * 不確実性が高いタスクのみSystem 2を起動し、エネルギー効率を最大化するメタ認知機能の実装。

## **⚡ Phase 3: The Rust Engine & Extreme Efficiency**

**目標:** Pythonの速度的限界を突破し、エッジデバイス（スマホ、ロボット）において**GPU不要・消費電力1/100**の推論を実現する。

### **🔑 Key Strategy: CPU Inference Optimization**

GPU（行列演算のモンスター）を使わず、CPU（分岐処理と整数演算が得意）で高速に動くエンジンを作ることで、ANNに対する「圧倒的な差別化」を図る。

* \[ \] **Rust Kernel Rewriting:**  
  * カーネルコア（event\_driven\_simulator.py）をRust言語で完全書き換え。  
  * Pythonオーバーヘッド（GIL）の排除。  
* \[ \] **SIMD Acceleration for Accumulation:**  
  * BitSpikeモデル特有の「足し算のみ（Accumulation）」の演算を、AVX-512/NEON命令を用いて爆速化。  
  * メモリアクセスの最適化（Cache-friendlyなデータ構造）。  
* \[ \] **Edge Device Deployment:**  
  * Raspberry Pi やスマートフォン上でのリアルタイム動作実証。  
  * **目標値:** 20W以下の消費電力で、実用的な対話エージェントを動作させる。

## **🌱 Phase 4: Structural Plasticity & Autonomy (The Living Brain)**

**目標:** 固定されたネットワークトポロジーからの脱却。学習＝「重みの変更」から「回路の組み替え」へ。

* \[ \] **Synaptogenesis & Pruning:**  
  * 共発火するニューロン間に物理的に新しいシナプスを生成し、使われないシナプスを削除する。  
  * これにより、ハードウェアリソースが許す限り「無限に学習し続ける」基盤を作る。  
* \[ \] **Sleep & Memory Consolidation:**  
  * 日中の短期記憶（海馬）を、睡眠中に長期記憶（皮質）へ転送・圧縮するサイクルの完全自動化。  
  * 夢（Generative Replay）による知識の汎化と整理。  
* \[ \] **Lifelong Learning:**  
  * 破滅的忘却（Catastrophic Forgetting）を防ぎ、新しいタスクを学習しても過去の知識を失わない能力の実証。

## **🔮 Phase 5: Hardware Embodiment**

**目標:** ソフトウェアの限界を超え、専用ハードウェア（FPGA/ASIC）による「物理的な脳」の実現。

* \[ \] **FPGA Accelerator:** SNNカーネルをハードウェア記述言語（Verilog/Chisel）に変換。  
* \[ \] **Neuromorphic Chip:** 非ノイマン型アーキテクチャによる、ミリワット級の超低消費電力AIチップの設計。

\#\#\# 修正のポイント

1\.  \*\*Phase 2の変更:\*\* 以前は「Rust化」でしたが、まずは\*\*「学習精度の壁」を突破する\*\*ことを優先し、BitNet \+ Mamba構成によるモデルアーキテクチャの確立をPhase 2に据えました。これにより「会話が成り立つレベル」を盤石にします。  
2\.  \*\*省エネ戦略の明確化:\*\* Phase 3で「Rust化」を行う目的を、単なる高速化ではなく\*\*「GPUを使わない（CPUだけで動く）ことによる省エネ」\*\*と定義しました。これがANNに対する最大の差別化要因（勝ち筋）となります。  
3\.  \*\*コンテキスト長への言及:\*\* SNNの弱点である記憶の長さを克服するため、Mamba（SSM）の導入を明記しました。

このロードマップに従うことで、\*\*「賢いが重いANN」\*\* に対し、\*\*「十分に賢く、圧倒的に軽くて省エネなDORA」\*\* というポジションを確立できます。  
