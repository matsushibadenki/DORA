# **DORA─SNN Project Roadmap: Neuromorphic Research OS─**

## **📌 プロジェクト方針の転換 (Strategic Pivot)**

本プロジェクトは、「特定のタスクを解くAIモデル」の開発から、**「知能現象を観測するための神経型計算基盤（Neuromorphic Research OS）」** の構築へと移行します。

### **基本哲学 (Core Philosophy)**

1. **機能ではなく現象**: 「意識」や「自由意志」をアルゴリズムとして実装するのではなく、複雑なネットワークダイナミクスから創発する**現象として観測**する。  
2. **汎用的な神経基盤**: Forward-Forward、STDP、Active Inferenceなど異なる学習則を、同一のスパイク神経網上で**プラグインのように差し替え・共存**可能にする。  
3. **時間発展としての知能**: 静的な推論ではなく、状態の時間発展と遷移を知能の本質として扱う。

## **📅 フェーズ 1: コアエンジンの統一 (The Kernel)**

**目標**: 異なる学習則やモデルを受け入れる「汎用ニューロン基盤」の確立。

* \[ \] **Universal Neuron Substrate (汎用ニューロン基盤) の設計**  
  * \[ \] SpikingLayer のリファクタリング: FF, STDP, Backpropを注入可能な構造へ  
  * \[ \] 状態変数（膜電位 v\_mem, スパイク履歴）の標準化  
  * \[ \] PlasticityRule インターフェースの策定（update\_weights() の抽象化）  
* \[ \] **学習則のモジュール化**  
  * \[ \] **Forward-Forward (FF)**: 視覚・特徴抽出向けモジュールとして整備（完了・統合）  
  * \[ \] **STDP (Spike-Timing-Dependent Plasticity)**: 連想記憶・海馬向けモジュール  
  * \[ \] **Active Inference**: 予測誤差最小化・運動制御向けモジュール  
* \[ \] **SNNエンジンの高速化**  
  * \[ \] 時間ループの行列演算化（In-place operations）  
  * \[ \] JITコンパイル / CUDAグラフ対応  
  * \[ \] スパース演算の導入

## **📅 フェーズ 2: ハイブリッド実験場の構築 (The Experiment)**

**目標**: 異なる学習則を持つ部位（モジュール）が、スパイク信号のみで連携するシステムの実証。

* \[ \] **異種混合ネットワークの実装**  
  * \[ \] **V1 (視覚野)**: Forward-Forwardで学習  
  * \[ \] **HC (海馬)**: STDPで学習  
  * \[ \] **Motor (運動野)**: Active Inferenceで制御  
  * \[ \] これらをスパイクバスで接続し、End-to-Endではなく局所学習のみで連携させる実験  
* \[ \] **Global Workspace (GWS) の実装**  
  * \[ \] モジュール間の長距離通信を担うハブ構造  
  * \[ \] 放送（Broadcast）と競合（Competition）のメカニズム実装  
* \[ \] **Sleep & Consolidation (睡眠と定着)**  
  * \[ \] オフライン学習フェーズの実装  
  * \[ \] FFのNegative Phaseとしての「夢」の活用

## **📅 フェーズ 3: 観測システムの構築 (The Observer)**

**目標**: ネットワーク内部の状態をリアルタイムに可視化・計測し、「心」の萌芽を探る。

* \[ \] **Consciousness Observer (意識観測器)**  
  * \[ \] **Φ (Phi) 計測**: 統合情報理論に基づく情報統合量の推定  
  * \[ \] **同期発火率 (Synchrony)**: ニューロン集団の同期イベント検知  
  * \[ \] **エントロピー解析**: 内部状態の複雑性と秩序のバランス計測  
* \[ \] **Research Dashboard (OS GUI)**  
  * \[ \] 脳波（LFP: Local Field Potential）のようなマクロ指標の表示  
  * \[ \] 概念形成の可視化（Concept Space Visualization）  
  * \[ \] 「迷い」「決断」の瞬間をログとして特定する機能

## **📅 フェーズ 4: ハードウェアと大規模化 (The Scale)**

**目標**: Neuromorphic Hardware上での動作と、文明規模のシミュレーション。

* \[ \] **Hardware Abstraction Layer (HAL)**  
  * \[ \] Intel Loihi / SpiNNaker 等のイベント駆動チップ向けコンパイラ  
  * \[ \] FPGA実装に向けたビット精度制約（Surrogate Gradientの排除）  
* \[ \] **Multi-Agent / Society Simulation**  
  * \[ \] 複数の "Brain OS" を搭載したエージェントによる社会実験  
  * \[ \] 言語・文化の創発シミュレーション

## **📂 (Legacy) 過去の実装・デモ**

*以下の機能は「OS」のモジュールまたは観測対象として再統合されます。*

* run\_spiking\_ff\_demo.py: → **Visual Cortex Module (FF)** として統合  
* run\_free\_will\_demo.py: → **Observer** による「自由意志現象」の観測へ移行  
* qualia\_synthesizer.py: → **Internal Representation Monitor** へ移行