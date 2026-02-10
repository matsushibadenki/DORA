# **SARA Engine Technical Specification**

**Current Version:** v35.1 (Production Release)

**Code Name:** Liquid Harmony

**Achievement:** MNIST Accuracy **95.10%** (CPU Only, No Backprop, No Matrix Multiplication)

## **1\. 概要**

SARA (Spiking Advanced Recursive Architecture) は、生物学的脳の「省電力・イベント駆動・自己組織化」を模倣した次世代AIエンジンです。

現代の深層学習（ANN）が依存する「誤差逆伝播法（BP）」や「行列演算」を完全に排除し、**スパースなスパイク通信のみ**で高度な認識・学習能力を実現しました。

## **2\. コア・アーキテクチャ (v35.1)**

### **A. Multi-Scale True Liquid Reservoir**

脳の皮質構造を模倣し、異なる時間特性を持つ3つのReservoir層を並列配置し、さらに\*\*層内再帰結合（Recurrent Connections）\*\*を実装しました。これにより、入力信号がなくなっても情報が内部で反響（Echo）し続ける「短期記憶」が生まれます。

| 層タイプ | ニューロン数 | 減衰率 (Decay) | 役割 | 再帰結合強度 |
| :---- | :---- | :---- | :---- | :---- |
| **Fast** | 1,500 | 0.3 (速い) | エッジ検出・ノイズ処理 | 1.2 (中) |
| **Medium** | 2,000 | 0.7 (中) | 形状・ストロークの統合 | 1.5 (強) |
| **Slow** | 1,500 | 0.95 (遅い) | 文脈・大域的パターンの保持 | 2.0 (最強) |

### **B. Adaptive Gain Homeostasis (適応的恒常性)**

ニューロンの発火率を一定範囲に保つための自律制御機構です。

* **仕組み**: 発火しすぎたニューロンは閾値を上げ、沈黙しているニューロンは閾値を下げます。  
* **Adaptive Gain**: 目標値との乖離が大きい場合、閾値の変更幅を動的に大きくすることで、急激な環境変化にも即座に適応します。

### **C. Sleep Phase (構造可塑性)**

学習エポック間に「睡眠フェーズ」を設け、不要な記憶を整理します。

* **Pruning**: 重みの絶対値が小さいシナプス（下位5%）を物理的に切断します。  
* **効果**: これにより過学習（Overfitting）を強力に抑制し、テスト精度が検証精度を上回る高い汎化性能を実現しました。

### **D. Momentum Delta Learning**

出力層（Readout）の学習には、BPを使わない以下の局所学習則を採用しています。

* ![][image1]**慣性項 (![][image2])**: 前回の更新量を加算することで、振動を抑えつつ収束を加速させます。  
* **クリッピング**: 勾配と重みを常に一定範囲に制限し、数値爆発（NaN）を完全に防いでいます。

## **3\. 開発の軌跡と技術的発見**

### **Phase 1: 迷走 (v13 \- v24) \- Acc 10%\~30%**

* **失敗**: 単純なSTDP（Hebbian）だけでは、数字のような複雑なパターンを分類できない。  
* **教訓**: 「特徴抽出」と「分類」を分ける必要がある。

### **Phase 2: 線形分離の限界 (v25) \- Acc 65%**

* **発見**: 教師ありDelta則を使えば学習は収束するが、隠れ層がない（単層）ため、線形分離不可能なパターンで詰む。

### **Phase 3: 静的Reservoirの失敗 (v26) \- Acc 8%**

* **失敗**: 画像を静的な値としてReservoirに入力しても、状態が変化せず情報が増えない。  
* **発見**: SNNには「時間変化」が必須である。

### **Phase 4: ポアソン符号化と安定化 (v27 \- v34) \- Acc 92%**

* **成功**: 入力をポアソン過程（確率的スパイク列）に変換することで、静止画を「動的な波紋」として処理することに成功。  
* **課題**: 隠れ層のニューロン同士がつながっていないため、情報の保持力が弱く、92%で頭打ちになった。

### **Phase 5: 真のLiquid State (v35.1) \- Acc 95.10%**

* **決定打**: \*\*再帰結合（Recurrent Connections）\*\*の導入。  
* **成果**: 内部でスパイクが相互作用することで、入力の「余韻」や「干渉」を含めた高次元な特徴表現が可能になり、人間の認識精度に肉薄した。

## **4\. 推奨パラメータ (Best Practice)**

MNISTタスクにおける黄金比率です。

* **Samples**: 20,000 (最低ライン)  
* **Reservoir Size**: 5,000 neurons (Fast:1500, Med:2000, Slow:1500)  
* **Input Scale**: Fast層には強く(1.0)、Slow層には弱く(0.4)入力する。  
* **Dropout**: 10% (汎化性能向上に必須)  
* **Sleep Pruning**: 5% (毎エポック)

graph TD  
    Image\[Image\] \--\>|Poisson Encoding| Spikes  
    Spikes \--\> Fast\[Fast Reservoir\]  
    Spikes \--\> Med\[Medium Reservoir\]  
    Spikes \--\> Slow\[Slow Reservoir\]  
      
    Fast \<--\> Fast  
    Med \<--\> Med  
    Slow \<--\> Slow  
      
    Fast \--\> Readout  
    Med \--\> Readout  
    Slow \--\> Readout  
      
    Readout \--\>|Momentum Delta| Class  
      
    subgraph "SARA Core"  
    Fast  
    Med  
    Slow  
    end  


[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAwAAAAZCAYAAAAFbs/PAAABZ0lEQVR4Xp1RsUoEMRTcVQThUATBFLfmJXLn1XqF/oBw2FxrZ2shWvgHdnK9IthYWWhjJYpYHFiJFv6NhZ6TS/KSze5aOFxe8ubNTJZclqXIpz8+V0nX2D6uPKoHS5tuKCEZxOYYro3Z6jmupUEtk/irypSqN8TWpjMzYeMaGxIEhnV51m63l5VSQynlGs8dSob1Xm9BKX0N8ZikPCKS50T0ZgKMpCiKLvkQIUQLwk+s+/5mf87nwPAK0YVJJEm3xvQfQ24GV1g/IDpebICAM0n0jc/bUFqNHamXIP7C4HlKRK+E1BPMJjA+4bzrDUM0ExhOvcG/GMSHxoB1x0F4vh1j0ErtOYqB5AN7g96O6DwH+YEbRiE7m4V4H0Hv0hoGxWrRARZ90hYGDzDeYI3QX+LmY+zzjn9B/9i1hvD/aa2EWBEt28W8JrQzjuHP+LsGv22453PCWyZsTdWixDQmNfMlKhYlBoNfy8NBMjzlSQIAAAAASUVORK5CYII=>