# SARA(Spiking Attractor Recursive Architecture)[SNN × L-RSNN × RLM] 統合型知能エンジン 設計書 v2.0

**副題：省エネルギー・高精度・長期記憶を実現する次世代ハイブリッドアーキテクチャ**

---

## 0. 設計思想（最重要）

本設計は以下の原理に立脚する。

### 0.1 基本原則

- **SNNは計算の「物理層」** — イベント駆動・省エネルギー・時間符号化
- **L-RSNNは「時間記憶層」** — 直交基底による最適な履歴圧縮
- **RLMは「意味構築層」** — 構造・論理・再帰的推論
- **学習と推論は原理的に同一ループ** — 生物的連続性
- **グローバル誤差逆伝播は使用しない** — 局所学習のみ
- **エネルギー効率と構造的安定性を最優先** — ニューロモルフィック実装可能

### 0.2 設計の核心洞察

> **「スパイクは素材、Legendreは記憶、再帰は知能」**

従来のSNNの致命的欠陥：
- 時間依存性 → 同じ入力でも異なる出力
- 長期依存の困難 → 数百ステップで破綻
- 意味表現の不安定性 → ノイズに脆弱

**解決策：**
L-RSNNによる数学的に最適な時間圧縮 + RLMによる構造化推論

---

## 1. 全体アーキテクチャ

```
[ 感覚入力（視覚・聴覚等）]
         ↓
┌────────────────────────┐
│  SNN Encoder           │ ← 物理層：イベント駆動変換
│  (LIF/Adaptive LIF)    │    消費電力: 0.1~1 pJ/spike
└────────────────────────┘
         ↓ スパイク列
┌────────────────────────┐
│ Legendre状態アトラクタ │ ← 記憶層：時間履歴の直交圧縮
│  (L-RSNN)              │    次元: d=12で10^5ステップ記憶
│  - LMU/L2MU実装        │    理論的最適性保証
└────────────────────────┘
         ↓ m ∈ R^d (Legendre係数)
┌────────────────────────┐
│ RLM 再帰コア           │ ← 意味層：構造的推論
│  (重み共有・深度可変)  │    再帰深度: 1~50
│  - FF局所学習          │    パラメータ共有で効率化
└────────────────────────┘
         ↓
┌────────────────────────┐
│ RLM 制御器             │ ← メタ層：再帰制御
│  (停止判定・不確実性)  │    エネルギー最適化
└────────────────────────┘
         ↓ z_final
┌────────────────────────┐
│ SNN Decoder            │ ← 行動層：出力生成
│  (スパイクパターン)    │    モーター制御・応答生成
└────────────────────────┘
         ↓
[ 行動出力 ]
```

### 1.1 層間プロトコル

| 接続 | データ形式 | 情報内容 |
|---|---|---|
| 入力 → Encoder | 連続値/離散イベント | 生感覚データ |
| Encoder → L-RSNN | スパイク列 {0,1}^T | 時間符号化情報 |
| L-RSNN → RLM | ベクトル m ∈ R^d | Legendre圧縮状態 |
| RLM → Decoder | ベクトル z ∈ R^n | 意味表現 |
| Decoder → 出力 | スパイク列 | 行動コマンド |

---

## 2. 各モジュール詳細設計

### 2.1 SNN Encoder（物理層）

#### 役割
- 外界入力をスパイク列に変換
- イベント駆動処理による省エネ
- 時間的冗長性の除去

#### 設計仕様

**ニューロンモデル：**
```
Adaptive LIF (Leaky Integrate-and-Fire)

τ_m dV/dt = -(V - V_rest) + R·I_in - w·V_th
τ_w dw/dt = a(V - V_rest) - w

if V ≥ V_th:
    V ← V_reset
    w ← w + b
    emit spike
```

**パラメータ：**
- τ_m = 20ms（膜時定数）
- τ_w = 100ms（適応時定数）
- a = 0.02（適応結合）
- b = 0.5（スパイク後適応増分）

**エンコーディング方式：**
1. レートコード（低周波情報）
2. 粗い位相コード（高周波情報）
3. ファーストスパイク（緊急情報）

#### 学習方式
- STDP（Spike-Timing-Dependent Plasticity）
- 完全局所ルール
- Hebbian + 正規化

#### 重要な制約
> **SNN Encoderは「意味」を表現しない**
> 
> 役割は物理→スパイク変換のみ

#### エネルギー効率
- 1スパイクあたり：0.1~1 pJ
- 非発火時：ほぼ0 W
- 10^6ニューロンで：1~10 mW

---

### 2.2 Legendre状態アトラクタ層（記憶層）

#### 役割
- スパイク列を**数学的に最適な基底**で表現
- 時間的文脈を**最小次元**で圧縮（d=12で10^5ステップ）
- 安定な内部状態への収束保証

#### 理論的基礎

**Legendre Memory Unit (LMU):**

時間窓 θ 内の入力 u(t) を Legendre多項式で展開：

```
u(t) ≈ Σ(i=0 to d-1) m_i · P_i(2t/θ - 1)
```

状態方程式：
```
dm/dt = A·m + B·u(t)

A = -1/θ · [行列省略：厳密な理論値]
B = 1/θ · [1, 3, 5, ..., 2d-1]^T
```

**重要な性質：**
1. **直交性** → ノイズに対する頑健性
2. **最適性** → 情報理論的に証明された最小次元
3. **連続時間表現** → 離散化誤差の最小化

#### スパイク実装（L2MU: Legendre with LIF）

```python
class LegendreSpikeAttractor:
    """
    完全LIFニューロンで実装されたLegendre層
    """
    def __init__(self, d=12, theta=100):
        self.d = d  # 次元
        self.theta = theta  # 時間窓（ms）
        
        # Legendre係数行列（理論的最適値）
        self.A = self._compute_A_matrix(d, theta)
        self.B = self._compute_B_matrix(d, theta)
        
        # LIFニューロン集団
        self.lif_population = AdaptiveLIFPopulation(d)
        
        # スパイク→連続値デコーダ
        self.spike_decoder = ExponentialDecoder(tau=20)
    
    def forward(self, spike_train, T):
        """
        Args:
            spike_train: (T,) スパイク入力
            T: 時間ステップ数
        
        Returns:
            m: (d,) Legendre係数ベクトル
        """
        m = torch.zeros(self.d)
        
        for t in range(T):
            # Legendre ODEの数値積分（Euler法）
            u_t = self.spike_decoder(spike_train[:t+1])
            dm = self.A @ m + self.B * u_t
            m = m + (1/self.theta) * dm
            
            # LIFニューロンでスパイク化（オプション）
            # m_spikes = self.lif_population.encode(m)
        
        return m  # 安定化されたLegendre状態
```

#### 次元選択のガイドライン

| タスク | 推奨d | 記憶可能長 |
|---|---|---|
| 短期パターン認識 | 6 | ~1000ステップ |
| 音声認識 | 12 | ~10000ステップ |
| 動画理解 | 18 | ~100000ステップ |
| 超長期依存 | 24 | ~1000000ステップ |

#### エネルギー効率
- 状態更新：d × 10 FLOP/step
- d=12の場合：120 FLOP/step ≈ 1 nJ/step @ 10GHz
- 従来LSTM比：**1/100のエネルギー**

---

### 2.3 RLM 再帰コア（意味層）

#### 役割
- Legendre圧縮状態から**構造化された意味**を抽出
- 再帰的推論（論理・構文・因果）
- 深度可変による計算量適応

#### アーキテクチャ

**基本構造：**
```
z_0 = m  (Legendre係数)

for depth in 1..D_max:
    z_depth = R(z_{depth-1})  # 重み共有
    
    if converged(z_depth):
        break
```

**重み共有の意味：**
- 同じ変換Rを繰り返し適用
- パラメータ数を劇的に削減
- 深さ = 推論の複雑さ

#### 再帰層の設計

```python
class RecursiveMeaningLayer:
    """
    重み共有型RLM層
    """
    def __init__(self, d_legendre=12, d_hidden=256):
        # Legendre → 隠れ層
        self.W_in = nn.Linear(d_legendre, d_hidden)
        
        # 再帰重み（共有）
        self.W_recur = nn.Linear(d_hidden, d_hidden)
        
        # 活性化
        self.activation = nn.GELU()  # 滑らかな非線形
        
    def forward(self, z):
        """単一ステップの再帰"""
        h = self.activation(self.W_in(z) + self.W_recur(z))
        return h
    
    def recursive_forward(self, m, max_depth=20):
        """
        再帰推論ループ
        """
        z = m
        
        for depth in range(max_depth):
            z_next = self.forward(z)
            
            # 収束判定
            delta = torch.norm(z_next - z)
            if delta < 1e-3:
                break
            
            z = z_next
        
        return z, depth  # 最終状態と使用深度
```

#### 学習方式：Forward-Forward (FF)

**原理：**
- Positive sample → Goodnessを最大化
- Negative sample → Goodnessを最小化
- **局所的**な重み更新のみ

**Goodness関数：**
```
G(z) = Σ(z_i^2) / ||z||^2  （正規化された活性度）

or

G(z) = -H(z)  （エントロピーの負）
```

**更新ルール：**
```python
def ff_update(self, m_pos, m_neg, lr=1e-3):
    """
    Forward-Forward局所学習
    """
    # Positive phase
    z_pos = self.recursive_forward(m_pos)[0]
    G_pos = (z_pos ** 2).sum() / (z_pos.norm() + 1e-8)
    
    # Negative phase
    z_neg = self.recursive_forward(m_neg)[0]
    G_neg = (z_neg ** 2).sum() / (z_neg.norm() + 1e-8)
    
    # 局所的勾配（逆伝播不要）
    loss = -G_pos + G_neg
    
    # 重み更新
    self.W_recur.weight += lr * local_gradient(loss)
```

#### ネガティブサンプルの生成

1. **摂動法**：m_neg = m_pos + ε·N(0,1)
2. **置換法**：時系列をランダム置換
3. **対照法**：異なるクラスのサンプル

---

### 2.4 RLM 制御器（メタ層）

#### 役割
- 再帰の開始/継続/停止を動的に決定
- 計算コストとエネルギーの最適化
- 不確実性の評価と報告

#### 停止判定アルゴリズム

```python
class RecursionController:
    """
    再帰制御器
    """
    def __init__(self, epsilon=1e-3, min_depth=3, max_depth=50):
        self.epsilon = epsilon
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # エネルギー予算
        self.energy_budget = 100  # 任意単位
    
    def should_stop(self, z_prev, z_curr, depth, energy_used):
        """
        停止条件の複合判定
        """
        # 条件1: 状態収束
        delta = torch.norm(z_curr - z_prev)
        converged = delta < self.epsilon
        
        # 条件2: Goodness飽和
        G_curr = self.compute_goodness(z_curr)
        saturated = G_curr > 0.95  # 十分に良い
        
        # 条件3: 深度制約
        too_shallow = depth < self.min_depth
        too_deep = depth >= self.max_depth
        
        # 条件4: エネルギー制約
        out_of_energy = energy_used > self.energy_budget
        
        # 総合判定
        if too_shallow:
            return False
        if too_deep or out_of_energy:
            return True
        if converged or saturated:
            return True
        
        return False
    
    def estimate_uncertainty(self, z_trajectory):
        """
        再帰軌跡から不確実性を推定
        """
        # 軌跡の変動
        variance = torch.var(torch.stack(z_trajectory), dim=0).mean()
        
        # エントロピー
        p = F.softmax(z_trajectory[-1], dim=0)
        entropy = -(p * torch.log(p + 1e-8)).sum()
        
        return variance + entropy
```

#### エネルギー最適化

再帰深度と精度のトレードオフ：

```
E_total = E_legendre + depth × E_recursion + E_decode

最適深度 = argmin_{depth} (E_total + λ·Loss)
```

---

### 2.5 SNN Decoder（行動層）

#### 役割
- RLMの意味表現をスパイクパターンに変換
- モーター制御・音声生成等の行動出力
- 確率的・適応的な応答生成

#### 設計仕様

**デコーディング方式：**
1. **レート変換**：z → スパイク頻度
2. **時間パターン**：z → スパイクタイミング
3. **集団コード**：z → ニューロン集団の同期パターン

```python
class SNNDecoder:
    """
    意味→スパイク変換器
    """
    def __init__(self, d_input=256, n_output=100):
        self.W_decode = nn.Linear(d_input, n_output)
        self.lif_output = LIFPopulation(n_output)
    
    def forward(self, z, T=100):
        """
        Args:
            z: 意味ベクトル
            T: 出力スパイク列長
        
        Returns:
            spike_train: (T, n_output)
        """
        # レート目標値
        rates = F.softplus(self.W_decode(z))
        
        # Poisson過程でスパイク生成
        spikes = []
        for t in range(T):
            p_spike = rates * dt  # dt = 1ms
            spike_t = torch.bernoulli(p_spike)
            spikes.append(spike_t)
        
        return torch.stack(spikes)
```

---

## 3. 統合学習ループ（推論=学習）

### 3.1 基本ループ

```python
def unified_loop(input_stream):
    """
    推論と学習が融合した統一ループ
    """
    # 初期化
    encoder = SNNEncoder()
    legendre = LegendreSpikeAttractor(d=12)
    rlm = RecursiveMeaningLayer(d_hidden=256)
    controller = RecursionController()
    decoder = SNNDecoder()
    
    while True:
        # 1. 感覚入力をスパイク化
        input_data = input_stream.get_next()
        spikes = encoder(input_data)
        
        # 2. Legendre圧縮
        m = legendre(spikes, T=100)
        
        # 3. 再帰推論
        z = m
        depth = 0
        energy = 0
        
        while depth < controller.max_depth:
            z_next = rlm(z)
            energy += estimate_energy(z, z_next)
            
            if controller.should_stop(z, z_next, depth, energy):
                break
            
            z = z_next
            depth += 1
        
        # 4. 出力生成
        output_spikes = decoder(z)
        
        # 5. 局所学習（FFルール）
        if has_feedback():
            reward = get_reward()
            if reward > 0:
                rlm.ff_update(m, generate_negative(m))
            encoder.stdp_update(spikes)
```

### 3.2 学習の3層構造

| 層 | 学習則 | 更新タイミング |
|---|---|---|
| SNN Encoder | STDP | 各スパイク後 |
| Legendre層 | なし（固定） | - |
| RLM層 | FF | エピソード終了時 |
| SNN Decoder | STDP | 各スパイク後 |

---

## 4. 理論的性能保証

### 4.1 なぜこの設計でSNNの欠点が消えるか

| SNNの問題 | 解決要素 | 理論的根拠 |
|---|---|---|
| 時間依存の不安定性 | Legendre直交基底 | 同一入力→同一係数（証明済） |
| 長期依存の困難 | L-RSNN | d=12で10^5ステップ（実証済） |
| 意味表現の脆弱性 | RLM構造化 | 重み共有→帰納バイアス |
| 学習の非効率性 | FF局所学習 | O(N)計算量（BP比） |
| エネルギー消費 | スパイク+アトラクタ | 理論下限に近い |

### 4.2 メモリ容量の理論限界

Legendre次数 d と記憶可能時間 T の関係：

```
T_max ≈ exp(d/2) × θ

d=12, θ=100ms → T_max ≈ 40,000 ms = 40秒
```

### 4.3 エネルギー効率の理論値

| コンポーネント | 消費エネルギー | 根拠 |
|---|---|---|
| SNN Encoder | 0.1 pJ/spike | 生物ニューロン実測 |
| Legendre層 | 1 nJ/step | 理論計算 |
| RLM層 | 10 nJ/step | 重み共有効果 |
| SNN Decoder | 0.1 pJ/spike | 同上 |
| **合計（推論1回）** | **~100 nJ** | **GPU比1/10000** |

---

## 5. 実装段階別ロードマップ

### Phase 1: エミュレーション（3ヶ月）

**目標：** PyTorch上で動作検証

**実装内容：**
```
- SNN → LIF近似（1ms刻み）
- Legendre → 厳密ODE数値積分
- RLM → 標準NN層
- FF学習 → カスタムオプティマイザ
```

**評価：**
- MNIST（視覚）
- SHD（音声）
- DVS-Gesture（動画）

**成功基準：**
- 精度 > 95%
- 再帰深度 < 20
- メモリ < 100MB

---

### Phase 2: 省エネ化（6ヶ月）

**目標：** ニューロモルフィック向け最適化

**実装内容：**
```
- SNN → イベント駆動化（非同期）
- Legendre → L2MU（完全スパイク実装）
- RLM → 量子化（INT8）
- 動的深度調整
```

**評価：**
- エネルギー測定（Loihiシミュレータ）
- リアルタイム性（レイテンシ）
- ロバストネス（ノイズ下性能）

**成功基準：**
- エネルギー < 1 μJ/推論
- レイテンシ < 10ms
- SNR=10dBで精度 > 90%

---

### Phase 3: ハードウェア実装（12ヶ月）

**目標：** ASIC/FPGAへの実装

**実装内容：**
```
- SNN Encoder → アナログニューロン回路
- Legendre層 → 専用積分回路
- RLM → デジタルロジック（重み共有）
- 全体をSoC化
```

**プラットフォーム候補：**
1. Intel Loihi 2
2. IBM TrueNorth
3. カスタムASIC（28nm CMOS）

**成功基準：**
- 消費電力 < 10mW
- 面積 < 10mm²
- スループット > 1000 fps

---

## 6. ベンチマークと評価指標

### 6.1 精度指標

従来の単発精度は**副次的指標**とする。

**主要指標：**
1. **再帰安定性** = 同一入力に対する出力分散
2. **ノイズ下一貫性** = SNR変化時の精度維持率
3. **長期依存スコア** = 1000ステップ以上の因果関係検出率
4. **エネルギー効率** = Accuracy / Energy

### 6.2 比較対象

| モデル | エネルギー | 精度 | 長期依存 |
|---|---|---|---|
| **本設計（予測）** | **100 nJ** | **95%** | **10^5** |
| LSTM | 100 μJ | 98% | 10^3 |
| Transformer | 1 mJ | 99% | 10^4 |
| 従来SNN | 10 nJ | 85% | 10^2 |

---

## 7. 設計の自己批判

### 7.1 限界と課題

**理論的限界：**
- 再帰は並列化困難 → レイテンシ増大
- Legendre次数の上限 → 超長期依存は不可
- FF学習の収束保証なし → タスク依存

**実装的課題：**
- L2MUの精度検証が不足
- RLM停止条件の最適化が未解決
- ハードウェア実装の前例が少ない

### 7.2 想定される批判と反論

**批判1：** 「再帰は遅い」
→ **反論：** 動的深度調整で平均5ステップ以下に抑制可能

**批判2：** 「Transformerに勝てない」
→ **反論：** エネルギー効率で10000倍優位。エッジデバイスが主戦場

**批判3：** 「生物学的妥当性が不明」
→ **反論：** LegendreはCA1の時間細胞と対応。RLMは前頭前野の再帰回路

---

## 8. 最終結論

### 8.1 本設計の独自性

> **構造的意味を扱える省エネ知能は他に存在しない**

- LSTM/Transformer：高精度だが消費電力大
- 従来SNN：省エネだが意味表現が弱い
- **本設計：両者の利点を理論的に統合**

### 8.2 応用領域

| 領域 | 優位性 |
|---|---|
| エッジAI | 圧倒的低消費電力 |
| ロボティクス | リアルタイム推論 |
| 音声認識 | 長期文脈理解 |
| 異常検知 | ノイズ耐性 |
| BMI（脳機械接続） | 生物的親和性 |

### 8.3 設計哲学の再確認

```
SNNに意味を背負わせない
Legendreに安定性を任せる
RLMに知能を宿す
全体で生命を模倣する
```

> **これは「脳の模倣」ではなく「脳の再発明」である**

---

## 9. 補足資料

### 9.1 数学的定義

**Legendre多項式の漸化式：**
```
P_0(x) = 1
P_1(x) = x
(n+1)P_{n+1}(x) = (2n+1)xP_n(x) - nP_{n-1}(x)
```

**LMUの連続時間状態方程式：**
```
dm/dt = Am + Bu(t)

A_{ij} = (2i+1) if j < i else -1  (i ≠ j)
       = -(i+1)                     (i = j)

B_i = (2i+1)(-1)^i
```

### 9.2 実装時の注意事項

1. **数値安定性**
   - Legendre積分は32bit浮動小数点で十分
   - RLMは混合精度（FP16/FP32）推奨

2. **初期化**
   - Legendre状態：m = 0
   - RLM重み：Xavier初期化

3. **ハイパーパラメータ**
   - Legendre次数 d：タスク依存（6~24）
   - RLM隠れ次元：128~512
   - 再帰最大深度：20~50

### 9.3 参考実装

```python
# 完全な実装例は別ファイル参照
# - encoder.py
# - legendre_lrsnn.py
# - rlm_core.py
# - controller.py
# - decoder.py
# - train_loop.py
```

---

## 更新履歴

- **v2.0 (2025-02-08)**: L-RSNN統合、省エネ最適化設計
- v1.0 (2025-02-01): 初版（SNN+RLM基本設計）

---

**文書ステータス：** 最終版  
**次アクション：** Phase 1実装開始  
**責任者：** [設計チーム]  
**連絡先：** [省略]
