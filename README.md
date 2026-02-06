# **DORA: Distributed Online Reactive Architecture 🧠**

**"The brain does not multiply matrices."**

DORAは、従来の深層学習（Deep Learning）の常識を覆す、**完全イベント駆動型（Event-Driven）** のスパイキングニューラルネットワーク（SNN）フレームワークです。

**行列演算（Matrix Multiplication）を一切行わず**、スパイク信号の非同期な伝播のみで学習・推論を行うことで、生物学的リアリティと圧倒的なエネルギー効率を両立させます。

*(DORA Kernel v1.5による脳活動のラスタープロット。青は興奮性、赤は抑制性ニューロンの発火を示す)*

## **🚀 Core Philosophy: The "No-Matrix" Manifesto**

現代のAIは、GPUによる巨大な行列演算（GEMM）に依存しています。しかし、生物の脳はそんな計算をしていません。DORAは以下の3つの制約（誓い）の下で開発されています。

1. **No Matrix Operations (行列演算の廃止):**  
   * ニューロンは ![][image1] を計算しません。シナプス前ニューロンからのスパイクを受け取った時だけ、膜電位を更新します（![][image2] complexity）。  
   * スパース（疎）な結合リスト（Adjacency List）を採用し、活動していない領域の計算コストはゼロです。  
2. **No Backpropagation (誤差逆伝播の廃止):**  
   * ニューラルネットワークを逆走する非生物的な学習則は使いません。  
   * **Predictive Coding (予測符号化)** と **STDP (スパイクタイミング可塑性)** を組み合わせ、局所的な「驚き（Surprise）」のみに基づいてシナプスを強化します。  
3. **No GPU Dependency (GPU依存からの脱却):**  
   * イベント駆動処理は本質的にシーケンシャルかつ非同期であり、CPUでの処理に適しています。  
   * 将来的にRust/C++による高速カーネルへ移行し、エッジデバイス（Raspberry Piなど）でも高度な知能を動作させることを目指します。

## **🛠 Architecture: DORA Kernel**

プロジェクトの中核は DORAKernel です。これはPyTorchモデルの構造（トポロジー）を読み込み、**物理的なニューロンとシナプスのグラフ**に変換（コンパイル）して実行します。

* **Event Queue:** 全てのスパイクは優先度付きキューで管理され、ミリ秒単位の正確なタイミングで処理されます。  
* **Dale's Law:** 興奮性（Excitatory）と抑制性（Inhibitory）のニューロンがバランスよく配置され（E/I Balance）、活動の暴走を防ぎつつ、豊かなリズム（脳波）を生み出します。  
* **Reverberation:** 入力が途絶えた後も、ニューロンの不応期と再帰結合により、短期記憶としての「残響」が維持されます。

## **📦 Installation & Usage**

### **1\. Setup Environment**

Python 3.10+ is recommended.

\# Clone repository  
git clone \[https://github.com/matsushibadenki/DORA.git\](https://github.com/matsushibadenki/DORA.git)  
cd DORA

\# Install dependencies  
pip install \-r requirements.txt

### **2\. Run the "Living Brain" Demo**

DORAの思考プロセスを可視化するデモを実行します。

python scripts/demos/brain/run\_brain\_v16\_demo.py

実行後、コンソールにスパイク処理の統計が表示され、runtime\_state/brain\_raster\_plot.png に脳活動のグラフが保存されます。

### **3\. Key Modules**

* snn\_research/hardware/event\_driven\_simulator.py: **The Heart.** イベント駆動カーネルの実装。  
* snn\_research/core/snn\_core.py: PyTorch APIとの互換性を保つブリッジレイヤー。  
* snn\_research/cognitive\_architecture/: 脳の各部位（前頭前野、海馬、基底核など）の実装。

## **🔮 Roadmap: Towards "Neuromorphic OS"**

現在のDORAはPythonプロトタイプですが、最終的には\*\*「OSそのものが脳である」\*\*システムを目指しています。

* **Phase 1 (Done):** Pythonによるイベント駆動カーネルの実証（v1.x）。  
* **Phase 2 (Next):** Rustによるカーネルの書き換えと高速化（100x speedup）。  
* **Phase 3:** 構造的可塑性（シナプスの物理的な生成・消滅）の実装。  
* **Phase 4:** Neuromorphic OSとしての統合（プロセス＝ニューロン群）。

## **🤝 Contribution**

"The brain is not a tensor processor."

この思想に共感する研究者・エンジニアの参加を歓迎します。

*Project DORA \- 2026 Matsushiba Denki Research*

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAADlklEQVR4Xr1UTYhNYRg+x8wwWAhxc8+d75zvzpW6LOSKJFlIzYLyk59RNqwMKwtFkY2/jd+URjREytDIgiQWsmKhTFMKg1nLQmIhM573nO+c853v59xzR3mad873Pe/z/nzvuedznH+FqxIpEpdNI/OuSWYkm6DlgCyKhEeaIkonX6YMIAfyJHKUTRO21HqEzEvIbJrB1k3Bwzh2r3wMm6Y5bP0lMJIRbC7bweQz5yCjoY3vs2Gf+Z9933+P9Sien2C8VCrNxHOEhXsGnz8aBMHbSqUynWKx3seYPyZiya5nM+f2OjmogYyxUyg8gWYOR+5UAX6QfLA+bNsShxPGbQP/Ac8GQkSQmr0AbFOXSZMAxQ9gchNo4FDW4zrgL0eH4r1qMHjyrcqy/xEovktM/LTMc85L4L+Lie+P2Kh7HLIOG0jV+tiMAyuiUWDj6ffaI5rrl3n8vvvBXSMf1kdjnhKBu+eVvYokT2GtlOuK4Ib9LMVQ1tUW1qYRVavV5oFbmQhicB6sEM0NxhwFU9NIsD30MXYu9fGegAfH4711nAURRzUajQ7UvQu7gZoXYcOoewzPq7Ah2FmphOvgpqixaOLPYhbBD7FfgMD1ovGBKMZtx/qJVy7PSDO0BpseNc/AdtK6Xq9PpboY0AvP8+biZvuN/fNUjSxofA6JYG9Egk1YH6ECXV1dy+nD5TgI+fzwQ2a7tV9n9GcE8vXCXuHAi1VfCBEYvlWRh7Sip72kwPMg53yREuNOgWMcDY2Jkz5FoU7yIkFVJHiJ9Wy8mcdJmAWJUzSB2Dsixx5JlgoMW2j7ohjGMwIVEH2D/aCToekdMU/NIngCUx8BfwH71Wox41rahm+UsQ2NxrKOlNfFMoMB3UcvX1LG0DQBzX0UU3kUc0Lqouk/4H/CbtvizVC6SdbG1unbOYkaW7BuE4O8FTvRw1bhywLka9g4xw1j8H2F/Qp4EKi+VqCfOWVwU60Vg+vDATbSd0UfK2mq1eos8ENYt2tJ6KaA82aGFCLw72AnVF6DMlEdNt5xuru759NNBjuPWpdgm2Gf0dcV2ANcEkvUmDAdHGvot5gwUg1MuqccX3/22hkUlGnApHm0cuk26kRPntgKTDZzCD3YmFeXORopB2bnZUYqaCp17JoilXToITl5bOdsGTmROa4E8oCLQxXbsiQDUAOyyPfKyFfKXpvSxhugHyZnGyE5sAIjqUCJlQcn/yf8BfU1rlVxW+ikAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAAEp0lEQVR4Xp2XW2gdRRjHzza13m+YEJvkzOwmkcCJIhiqtUpBfJIiKBQfRMW2iiKiIiJKvRR8EKVaFH0RH2zVF0EQFcEL2IdSrbUPFS1SWiraFmuRoH2QWNv4+3Zmdr+d3T2J/mGyO//v/13muiedjkcSXgpopu29jv5WDadcfGQNUf4nzwUFEZLFubRpCr5NIPgfA2iULZisjddYjKYfGgfTp2hgjLl4bGzsbBH0Vzo0ahIXZ2Rk5JyqYZGYnJw8M8uy1dbaNbTlsT0eWLfbHSHhrvHx8QuVpUTbGBoGSZyeNXbH8PDwuQW5ECh2Kk3TD601+3huIsjzFP4nz0/TLBsOOl2HDBLN12mWritIXxAxLsL2I8/9aZrtd0//noV319C9FCLLu9TB65IiZjMSSfIkDr9S5F0QS0NxDAba7qYdGhoaOk+55GCQm5mh3Y6JlgL0er1l+D6NZp7nc+Q5a2Zm5gzZVqOjo5fIqhprjxpjtwQvJuMC9Meo5d56TNVHsIUAc1mWXl2yJUh2N/Z5a8yzgSsSsCK0VUpeIKTA/z2Km+e5UvPl4O3L2B7RJPr11PVzb7q3zKu1OQ96S16Uta9XBAqyh73mW+k730QG/DDc99VZcdAMmmO0E53qSl4psy9KbBsp9Nag95oB+L9t4HVAOUwYjtD+6o51R+rpHaamps5nS5xGN1uQSV7Q5yzxB6UyipD4w+YG/YnWUOhOOR/yju12BpEGt6BJ3d5/reS9DfJRCmIL2Dc0HwPdKp/8YMQfoG3WXAzsD3jfF5lldpuZoaBX6H8ZazWkDBksbW/JeEB+4YPeU7KCavUkus/rPlJ0vpTY7o/1gsDI/hZfnt8wQTt53yd9U5yXpOquuuheZWWOVwRysjHM+aDXxM4aUrDXFQdIZs8VYG+sqivuCZrfaH/wPqAK2o7/aqfI+9G3wikZ3EPYTnWKazEp7l/Zt6eHBsM1V585nKe97nA4TAL6V0nh8oxcHJL8UF+ea4z9uCCBrIJck9LlWryC/lblWYDcd0ruadHq2iAPSmC5NXKisLkX+cvHYqtoJIh2hpMpF35tQUbA/qDc3yz3Y7EtAM27tGs1F7IQ+xlsR7QtB0G3+eS3xTYBRa/D/o8sWbwa7qaR7ZM9UbJJZfDEf99PzApvLWQCbDfQ9gRamXJgoz6zQ1GdXIFhOW2W9pV8xfRM47AB/qT7enkyAprDaN7UXJD5L+bv1t3fA1ozyBdYBmzdGdug80arKod5W0GUSDrc3yuyNP0OwQGZWfbbC6n7/fAZ/etjD4Fayrfw265J/zX9wU/IPO0U7RfaIdpPNv9umJOeP5r/omyBlYNtzON58IaJEyyV+5V2B+I1GTdGLMgRORN0LUUcd4e2AfHeqEHzVc3ExESXWuaIfWnFUKIetM60Yol19/LGKh3NkB5AFLwtl3GH9h15r2hcJ/rbmKwObWKVbibBCZZ8VNHNaCy6noiY1xFzlpiXOaY+4Ar0AOpB2z1J8hTt7ZiPXdojVCCrKD+jb4oNDm1BGwdXZ2KwFzfxL9dgzAc0RoiSSpc4Kyl6vVK1oH/EqFvnHVPn9XtjCo0FBSX+BVFyCqMRUm+pAAAAAElFTkSuQmCC>