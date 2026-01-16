# ─ DORA Project Roadmap: Neuromorphic Research OS (Revised) ─

---

## 📌 プロジェクト方針の転換（確定）

本プロジェクトは、
**タスク最適化AIの開発ではなく**  
**知能現象を観測・比較・再現するための神経型計算OS** を構築する。

本OSは「答えを出すAI」ではなく、

> **知能が立ち上がる条件を実験できる計算基盤**

である。

---

## 🧠 基本哲学（不変）

1. **機能ではなく現象**
   - 意識・自由意志・自己は「実装」しない
   - 神経ダイナミクスから**創発する現象として計測**する

2. **学習則は差し替え可能な部品**
   - Forward-Forward / STDP / Active Inference を
     同一スパイク基盤上で**共存・比較可能**にする

3. **時間発展こそが知能**
   - 静的推論は禁止
   - 全ての処理は「状態遷移」として記述される

---

## 📅 Phase 1: Universal Neuron Kernel（The Kernel）

### 目的
**あらゆる学習則を載せられる「神経OSの核」**を確立する。

> このフェーズでは「賢さ」は問わない  
> **破綻しない・差し替えられる・観測できる**ことが成功条件

---

### 1. Universal Neuron Substrate（最優先）

- [ ] **Neuron / Synapse / Spike を最小単位として再定義**
- [ ] 膜電位・閾値・回復変数・スパイク履歴の**標準状態モデル化**
- [ ] 時間ステップではなく **イベント駆動更新**を原則とする
- [ ] 行列・バッチ前提APIの完全排除

---

### 2. PlasticityRule Interface（中核API）

- [ ] `PlasticityRule.update(pre_spike, post_spike, local_state)`
- [ ] **誤差・勾配・global loss を引数に取らない**
- [ ] 学習則は「観測できる副作用」として実装

対応ルール：

- [x] Forward-Forward（局所正負フェーズ）
- [ ] STDP / R-STDP
- [ ] Active Inference（予測誤差＝内部状態差分）

---

### 3. 実行エンジンの原則

- [ ] Python / NumPy 1.x（疎データ構造中心）
- [ ] for-loop最適化は許可（**意味論優先**）
- [ ] JITは「後段最適化」としてのみ使用
- [ ] GPU / CUDA 依存は**禁止**

---

## 📅 Phase 2: Heterogeneous Brain Experiment（The Experiment）

### 目的
**異なる学習原理を持つ領域が、スパイクのみで協調できるか**を検証する。

> End-to-End最適化は禁止  
> 「局所最適 × 相互作用」のみで成立するかを見る

---

### 1. 異種混合ネットワーク（必須）

- [ ] **V1（感覚野）**
  - Forward-Forward
  - 高頻度入力・高速適応

- [ ] **Hippocampus（記憶）**
  - STDP / R-STDP
  - 連想・再生・再固定化

- [ ] **Motor / Action**
  - Active Inference
  - 誤差 = 予測と観測の差分

- [ ] **Spike Bus**
  - 全通信はスパイクのみ
  - 値の直接共有は禁止

---

### 2. Global Workspace（統合だが支配しない）

- [ ] 長距離スパイク通信ハブ
- [ ] Broadcast（同報）と Competition（競合）
- [ ] **意思決定器ではない**
- [ ] 活性化は一時的・揮発性

---

### 3. Sleep & Consolidation（必須）

- [ ] オフラインフェーズの明示的導入
- [ ] 入力遮断状態での内部再活性
- [ ] FFの Negative Phase を「夢」として利用
- [ ] 学習は**覚醒より睡眠で進む**構造を許可

---

## 📅 Phase 3: Observer Layer（The Observer）

### 目的
**「何が起きているか」を測れなければ、知能研究ではない**

---

### 1. Consciousness / Dynamics Observer

実装対象ではなく**計測器**として扱う。

- [ ] **統合度指標（Φ近似）**
- [ ] 同期発火イベント検出
- [ ] 状態遷移エントロピー
- [ ] 活性分布の自己相関

※ 数値は**真理ではなく比較指標**

---

### 2. Research Dashboard（OSとしての顔）

- [ ] LFP風マクロ信号表示
- [ ] 活性クラスタの時系列追跡
- [ ] 「迷い」「切替」「破綻」イベントの自動抽出
- [ ] 再現実験ログ（seed / 初期条件）

---

## 📅 Phase 4: Scale & Hardware（The Scale）

### 目的
**このOSは机上の空論ではない**ことを示す。

---

### 1. Hardware Abstraction Layer（制約前提）

- [ ] Loihi / SpiNNaker 向けイベントマッピング
- [ ] オンチップ学習前提（BP不可）
- [ ] ビット精度・遅延制約を**最初から受け入れる**
- [ ] FPGA実装は「理論検証用」

---

### 2. Multi-Agent / Society Simulation

- [ ] 複数 Brain OS の相互作用
- [ ] 教師なし模倣・文化伝播
- [ ] 言語・規範・役割の創発観測
- [ ] **中央制御・グローバル最適化は禁止**

---

## 📂 Legacy Code の再定義

過去コードは「失敗」ではなく  
**観測対象として再統合**する。

- `run_spiking_ff_demo.py`
  → Visual Cortex（FF）標準実装

- `run_free_will_demo.py`
  → 意思決定遷移の観測シナリオ

- `qualia_synthesizer.py`
  → 内部表現モニタ（Observer）

---

## 🎯 最終的な成功条件

このプロジェクトの成功とは：

- AGIを名乗ることではない
- 人間らしい振る舞いを演出することでもない

> **制約を守ったまま、  
> 知能現象が「確かに起きた」と  
> 観測・再現・比較できること**

である。
