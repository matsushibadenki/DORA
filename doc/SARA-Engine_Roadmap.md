# **SARA Engine Roadmap**

**Mission:** To build a biologically plausible, CPU-efficient, general-purpose intelligence engine without Backpropagation.

## **✅ Phase 1: Genesis (視覚野の獲得)**

**Status: Completed (v35.1)**

* **目標**: MNISTにおいて、BPなしで95%以上の精度を達成する。  
* **成果**:  
  * 再帰型Reservoir Computing (LSM) の確立。  
  * Homeostasisによる自律安定化。  
  * Sleep Phaseによる構造可塑性の実証。  
  * **Acc: 95.10% achieved.**

## **🚀 Phase 2: Cognition (言語野の獲得)**

**Target: SARA v40.0 \- v49.0**

画像（空間）だけでなく、テキストや音声（時間）を理解する能力を獲得します。

### **Step 2.1: Sequence Prediction (v40 \- v42)**

* **タスク**: 文字列（"A", "B", "C"...）を入力し、次に来る文字を予測する。  
* **技術**:  
  * スパースな単語埋め込み（Sparse Embedding）。  
  * 長期記憶（Slow Reservoir）の強化。  
  * 文脈窓（Context Window）の拡張。

### **Step 2.2: Liquid Language Model (LLM) (v43 \- v45)**

* **タスク**: 簡単な文章生成、感情分析。  
* **技術**:  
  * **Hierarchical Reservoir**: 下層が単語を、上層が文脈を処理する階層構造。  
  * **Concept Cells**: 特定の単語や意味に反応する概念ニューロンの形成。

### **Step 2.3: Multi-Modal Integration (v46 \- v49)**

* **タスク**: 画像とテキストの関連付け（"7"の画像を見て "Seven" と答える）。  
* **技術**:  
  * 視覚野（Phase 1）と言語野（Phase 2）の連合野（Association Area）による統合。

## **🤖 Phase 3: Embodiment (運動野・小脳の獲得)**

**Target: SARA v50.0 \- v59.0**

認識するだけでなく、環境に働きかける能力（制御）を獲得します。

* **v50**: PID制御の代替（倒立振子など）。  
* **v53**: 強化学習（R-STDP）の再統合。報酬に基づく行動最適化。  
* **v55**: ロボットアーム制御、ドローン制御シミュレーション。

## **⚡ Phase 4: Systematization (ハードウェア化・OS化)**

**Target: SARA v60+**

Pythonの枠を超え、実システムとして稼働させます。

* **Rust Porting**: PythonコードをRust（dora\_kernel）へ完全移植し、10〜100倍の高速化を実現。  
* **FPGA/ASIC Design**: SARAのスパース演算に特化した回路設計。  
* **Neuromorphic OS**: SARAをカーネルとした、学習し続けるオペレーティングシステム。

## **現在の注力領域 (Current Focus)**

**Transitioning from Phase 1 to Phase 2**

* \[ \] テキストデータをスパイク信号へ変換するエンコーダーの開発。  
* \[ \] 時系列データの記憶保持能力（Memory Capacity）のベンチマーク。  
* \[ \] Fashion-MNIST等、より複雑な画像タスクでの検証。