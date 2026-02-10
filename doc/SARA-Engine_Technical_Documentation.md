# **SARA Engine Technical Documentation**  
[SNN × L-RSNN × RLM] 統合型知能エンジン  

**Current Version:** v35.1 (Production Release v1.0)

**Code Name:** Liquid Harmony

**Achievement:** MNIST Accuracy 95.10% (No Backprop, No Matrix Multiplication, CPU Only)

## **1\. プロジェクト概要**

SARA (Spiking Advanced Recursive Architecture) Engineは、生物学的制約に基づいた次世代の人工知能エンジンです。

現代のディープラーニングが依存する「誤差逆伝播法（Backpropagation）」や「行列演算（Matrix Multiplication）」、「GPUによる力任せの並列計算」を一切使用せず、**スパースなイベント駆動型計算**のみで高精度な学習を実現することを目的としています。

### **コアフィロソフィー**

1. **Pure Spiking**: すべての情報はスパイク（0/1のイベント）のタイミングと頻度で表現される。  
2. **Bio-Plausibility**: 局所的な学習則（STDP/Delta）、恒常性（Homeostasis）、構造可塑性（Sleep Phase）のみを用いる。  
3. **CPU Optimized**: 行列演算を排除し、インデックス参照によるスパース計算でCPUでの高速動作を実現する。

## **2\. アーキテクチャの進化 (The Road to 95%)**

10%程度のランダムな精度から95%に至るまでの、主要な技術的転換点（Paradigm Shift）の記録です。

### **Phase 1: 迷走と停滞 (v13 \- v24)**

* **アプローチ**: STDP（教師なし学習）による特徴抽出とR-STDP（強化学習）による分類。  
* **結果**: 精度 10%〜30%。  
* **失敗の原因**:  
  * **表現力の欠如**: 単純なLIFニューロンではMNISTのパターンを分離しきれなかった。  
  * **教師信号の弱さ**: R-STDPは収束が遅く、探索空間が広すぎた。  
  * **競合の不全**: 側方抑制がうまく機能せず、特徴が平均化してしまった。

### **Phase 2: 線形分離の限界 (v25)**

* **アプローチ**: **教師ありDelta則**（Widrow-Hoff学習）をSNNに直接適用。  
* **結果**: 精度 65.8%。  
* **発見**: BPを使わなくても、エラー訂正学習（Delta Rule）を行えば学習は収束する。しかし、隠れ層なしの単層構造では線形分離不可能なパターンを解けない。

### **Phase 3: Reservoir Computingへの転換 (v26 \- v28)**

* **アプローチ**: 固定された巨大なランダム結合層（Reservoir）で入力を高次元空間へ射影し、Readout層のみを学習させる「Liquid State Machine (LSM)」構成へ移行。  
* **結果**:  
  * **v26 (Static)**: 精度 8%（入力が静的すぎてReservoirが活動しない）。  
  * **v27 (Poisson)**: **ポアソン符号化**により入力を時系列化。精度向上。  
  * **v28 (Resonance)**: スペクトル半径の調整とMomentum学習の導入。しかし発火率の制御に失敗（暴走 or 死滅）。

### **Phase 4: 安定化と汎化 (v29 \- v34)**

* **アプローチ**:  
  * **Per-Sample Update**: 毎ステップではなく、エピソード（画像1枚）ごとの勾配蓄積更新。  
  * **Sleep Phase**: 学習後に不要なシナプスを刈り込む（Pruning）構造可塑性。  
  * **Multi-Scale**: Fast/Medium/Slowの異なる時定数を持つReservoirを並列化。  
* **結果**: 精度 92.0%。  
* **課題**: 隠れ層のニューロン同士がつながっていないため、情報の「反響」が起きず、記憶容量に限界があった。

### **Phase 5: 完成 (v35 \- v35.1)**

* **アプローチ**: **再帰結合 (Recurrent Connections)** の実装。  
* **結果**: 精度 **95.10%**。  
* **決定打**:  
  * 隠れ層内でスパイクが相互作用し、入力がなくなった後も情報が保持される「真のLiquid State」を実現。  
  * **Adaptive Gain Homeostasis**: 発火率に応じて閾値を動的に調整し、全ニューロンを有効活用。

## **3\. 採用された主要技術 (Core Technologies)**

### **A. Multi-Scale True Liquid Reservoir**

脳の皮質構造を模倣し、異なる時間特性を持つ3つのReservoir層を並列に配置し、内部で再帰的に接続しています。

| Layer Type | Decay | Time Scale | Role | Recurrent Scale |
| :---- | :---- | :---- | :---- | :---- |
| **Fast** | 0.3 | 短期・鋭敏 | ノイズ、エッジ検出、急激な変化 | 1.2 (Strong) |
| **Medium** | 0.7 | 中期 | ストローク、形状の断片 | 1.5 (Very Strong) |
| **Slow** | 0.95 | 長期・持続 | 文脈、大域的な形状 | 2.0 (Dominant) |

* **Recurrent Connections**: これにより、静止画であるMNISTを「動画」のように動的なパターンとして処理可能にしました。  
* **Spectral Initialization**: 重み行列のスペクトル半径を調整し、信号が減衰も爆発もしない「クリティカル（臨界）状態」を維持します。

### **B. Homeostasis (恒常性維持機構)**

ニューロンが「生き残る」ために閾値を自律調整するメカニズムです。

* **過活動抑制**: 発火しすぎたニューロンは閾値を上げ、情報の飽和を防ぐ。  
* **死滅防止**: 発火しないニューロンは閾値を下げ、強制的に学習に参加させる。  
* **Adaptive Gain**: 目標発火率との乖離が大きい場合、閾値の変更幅を大きくして素早く追従させる。

### **C. Sleep Phase (構造可塑性)**

生物の睡眠を模したフェーズをエポック間に挿入します。

* **Pruning (刈り込み)**: 重みの絶対値が小さいシナプス（下位5%など）を物理的に切断（ゼロ化）します。  
* **Weight Decay**: 全体の重みをわずかに減衰させ、特定の結合だけが肥大化するのを防ぎます。  
* **効果**: 過学習（Overfitting）を強力に抑制し、テスト精度が検証精度を上回る理想的な汎化性能を実現しました。

### **D. Momentum Delta Learning**

Readout層（出力層）の学習には、BPを使わない以下の局所学習則を採用しています。

* ![][image1]エラー訂正（Delta Rule）に慣性項（Momentum）とAdagrad的な学習率調整を組み合わせ、高速かつ安定した収束を実現しました。  
* **クリッピング**: 勾配と重みを常に一定範囲（\[-3, 3\]等）に制限し、数値爆発（NaN）を完全に防いでいます。

## **4\. システム構成**

graph TD  
    Input\[Input Image (784)\] \--\> Poisson\[Poisson Encoder\]  
    Poisson \--\> Fast\[Fast Reservoir (1500)\]  
    Poisson \--\> Med\[Medium Reservoir (2000)\]  
    Poisson \--\> Slow\[Slow Reservoir (1500)\]  
      
    Fast \<--\> Fast  
    Med \<--\> Med  
    Slow \<--\> Slow  
      
    Fast \--\> Readout\[Readout Layer (10)\]  
    Med \--\> Readout  
    Slow \--\> Readout  
      
    Readout \--\> Output\[Classification\]  
      
    subgraph "Learning & Plasticity"  
    Homeostasis \-.-\> Fast & Med & Slow  
    SleepPhase \-.-\> Readout  
    DeltaRule \-.-\> Readout  
    end

## **5\. 今後の展望 (Next Steps)**

現在のエンジンはMNISTに対して「完成」しましたが、汎用知能への道のりは続きます。

1. **Fashion-MNISTへの適用**:  
   * より複雑なパターン認識能力の検証。  
2. **ハードウェア実装**:  
   * 現在のPython実装は概念実証です。Rust/C++による高速化、あるいはFPGA/ASICへの焼付けが可能です。  
3. **連続学習 (Continual Learning)**:  
   * 「新しいタスクを学習しても、古いタスクを忘れない」能力の強化。SARAのSleep Phaseはこの特性と非常に相性が良いです。

[image1]: <data:image/png;base64,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>