# **DORA Project Structure & Architecture 🏗️**

DORAは、単なるAIモデルではなく、**「脳を模倣したオペレーティングシステム（Neuromorphic OS）」** として設計されています。

ディレクトリ構造も、生物学的な階層性と機能分担を反映しています。

## **📂 Directory Overview**

### **snn\_research/ (Core Research Modules)**

脳の本体コードです。

* **hardware/ (The Physical Layer)**  
  * event\_driven\_simulator.py: **最重要ファイル。** 行列演算を使わないスパイクシミュレータ（DORA Kernel）。ニューロン、シナプス、イベントキューを管理。  
  * compiler.py: 将来的なFPGA/Neuromorphic Chipへのマッピング用。  
* **core/ (System Integration)**  
  * snn\_core.py: SpikingNeuralSubstrate。PyTorchで定義された抽象的なモデルを、DORA Kernelの物理グラフに変換するブリッジ。  
  * neuromorphic\_os.py: 脳活動をスケジュールし、リソース（エネルギー、CPU時間）を管理するOSカーネル。  
  * neurons/: LIF (Leaky Integrate-and-Fire) などのニューロンモデル定義。  
* **cognitive\_architecture/ (Functional Areas)**  
  * 脳の各部位をモジュール化したもの。  
  * prefrontal\_cortex.py: 意思決定、計画（PFC）。  
  * hippocampus.py: エピソード記憶、短期記憶。  
  * basal\_ganglia.py: 行動選択（強化学習）。  
  * global\_workspace.py: 意識の座（情報の放送局）。  
  * astrocyte\_network.py: エネルギー供給と恒常性維持。  
* **learning\_rules/ (Plasticity)**  
  * predictive\_coding\_rule.py: 予測誤差に基づく局所学習。  
  * stdp.py: スパイクタイミング依存可塑性。

### **scripts/ (Execution & Experiments)**

実験やデモを実行するためのスクリプト群。

* **demos/brain/**: 統合された脳の動作デモ。  
  * run\_brain\_v16\_demo.py: 最新のイベント駆動カーネルを使ったメインデモ。  
* **experiments/**: 特定の仮説検証用の実験スクリプト。

## **🧠 Architectural Layers**

DORAのアーキテクチャは以下の3層構造になっています。

### **Layer 1: The Event Kernel (Micro-Level)**

* **実体:** DORAKernel  
* **役割:** スパイク（電気信号）の物理シミュレーション。  
* **特徴:** 行列なし。時間駆動ではなくイベント駆動。ミリ秒単位の精度。

### **Layer 2: The Cognitive Modules (Meso-Level)**

* **実体:** ArtificialBrain クラス内の各コンポーネント（PFC, Hippocampus等）。  
* **役割:** 機能的な情報処理。  
* **特徴:** 各モジュールはニューロンの集まり（Neuron Group）としてKernel上に定義される。

### **Layer 3: The Neuromorphic OS (Macro-Level)**

* **実体:** NeuromorphicOS  
* **役割:** システム全体の統合とリソース管理。  
* **特徴:** 外部入力（カメラ、テキスト）をスパイクに変換し、脳に投入。脳の出力を解釈し、行動（APIコール等）に変換。

## **🔄 Data Flow (The Cognitive Cycle)**

1. **Sensation:** 外部データ → SpikeEncoder → スパイク列。  
2. **Transmission:** カーネル内のイベントキューを経由して、Sensory Cortexへ到達。  
3. **Perception:** Sensory Cortexでの特徴抽出（Predictive Coding）。  
4. **Consciousness:** 顕著な情報は GlobalWorkspace に上がり、全脳へ放送される。  
5. **Decision:** PrefrontalCortex が計画し、BasalGanglia がアクションを選択。  
6. **Action:** MotorCortex が出力を生成 → 外部へ。

このサイクルが、クロック（行列演算ループ）ではなく、\*\*「波（Wave）」\*\*として非同期に伝播するのがDORAの最大の特徴です。