# **DORA: Distributed Online Reactive Architecture 🧠**

**"The brain does not multiply matrices."**

*(脳は行列演算を行わない)*

**DORA** は、現代の深層学習（Deep Learning）の常識を覆す、**完全イベント駆動型（Event-Driven）** のスパイキングニューラルネットワーク（SNN）フレームワークです。

我々は、「行列演算（Matrix Operations）」「誤差逆伝播（Backpropagation）」「GPU依存」という現代AIの3大要素を**意図的に排除**しました。代わりに、生物学的な制約（スパイク通信、局所学習、遅延）を取り入れることで、真に効率的で自律的な知能の実現を目指しています。

*(DORA Kernel v1.5による脳活動のラスタープロット。青は興奮性、赤は抑制性ニューロンの発火を示す)*

## **🚀 Core Philosophy: The "No-Matrix" Manifesto**

DORAは、以下の3つの厳格な制約（誓い）の下で設計されています。

### **1\. No Matrix Operations (行列演算の廃止)**

* **現状のAI:** 全結合層では ![][image1] の行列積（GEMM）を行い、0（スパイクなし）の要素に対しても無駄な計算リソースを費やしています。  
* **DORA:** **隣接リスト（Adjacency List）** 構造を採用。ニューロンはスパイクを受け取った瞬間だけ起動し、接続先のみに信号を送ります。計算量は発火数 ![][image2] に比例する ![][image3] となり、スパースな脳活動において圧倒的な効率を誇ります。

### **2\. No Backpropagation (誤差逆伝播の廃止)**

* **現状のAI:** ネットワーク全体を逆走して微分を連鎖させるBP法は、生物学的にあり得ないメカニズムであり、膨大なメモリを消費します。  
* **DORA:** **Predictive Coding (予測符号化)** と **STDP (スパイクタイミング可塑性)** を採用。各シナプスは、局所的な「予測誤差（Surprise）」と「因果関係」のみに基づいて、自律的に結合強度を調整します。

### **3\. No GPU Dependency (GPU依存からの脱却)**

* **現状のAI:** 大規模並列演算のために高価で電力食いなGPUが必須です。  
* **DORA:** イベント駆動処理は本質的にシーケンシャルかつ非同期です。DORAはCPU（シングルスレッドまたはマルチコア）で高速に動作するように最適化されており、将来的にはRaspberry Piなどのエッジデバイスで「生きた脳」を動かすことを目指します。

## **🛠 Architecture: The DORA Kernel**

プロジェクトの中核は snn\_research/hardware/event\_driven\_simulator.py に実装された **DORA Kernel** です。

* **Compiler:** PyTorchで定義された抽象的なモデル構造（nn.Linearなど）を読み込み、物理的なニューロンとシナプスのグラフに「コンパイル」します。  
* **Event Queue:** 全てのスパイクイベントは優先度付きキューで管理され、ミリ秒単位の正確なタイミングで非同期に処理されます。  
* **E/I Balance:** **Dale's Law（デール則）** に基づき、興奮性（Excitatory）と抑制性（Inhibitory）のニューロンをバランスよく配置。活動の暴走（てんかん状態）を防ぎ、安定したリズムを生み出します。

## **📦 Quick Start**

### **1\. Installation**

Python 3.10+ 環境が必要です。

\# Clone repository  
git clone \[https://github.com/matsushibadenki/DORA.git\](https://github.com/matsushibadenki/DORA.git)  
cd DORA

\# Install dependencies  
pip install \-r requirements.txt

### **2\. Run the Demo**

DORAの思考プロセス（発火の伝播と抑制）を可視化するデモを実行します。

python scripts/demos/brain/run\_brain\_v16\_demo.py

実行後、コンソールにスパイク処理の統計（OPS）が表示され、runtime\_state/brain\_raster\_plot.png に脳活動の画像が生成されます。

## **📂 Key File Structure**

* snn\_research/hardware/event\_driven\_simulator.py: **The Heart.** 行列を使わないイベント駆動カーネルの実装。  
* snn\_research/core/snn\_core.py: **The Bridge.** PyTorch APIとDORA Kernelをつなぐブリッジ。  
* snn\_research/core/neuromorphic\_os.py: **The OS.** 脳をOSとして管理するスケジューラ。  
* snn\_research/cognitive\_architecture/: 脳の各機能モジュール（PFC, Hippocampus, etc.）。

## **🔮 Future Roadmap**

現在の実装はPythonによるプロトタイプ（Phase 1）です。

* **Phase 2:** カーネルの **Rust** による書き換え（100倍以上の高速化）。  
* **Phase 3:** **構造的可塑性**（シナプスの物理的な生成・消滅）の実装。  
* **Phase 4:** Neuromorphic OSとしての完全統合。

詳細なロードマップは doc/ROADMAP.md を参照してください。

*Project DORA \- Matsushiba Denki Research 2026*

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADgAAAAYCAYAAACvKj4oAAAFHUlEQVR4XrWXe4gVVRzH75SRQi+ppdx77zzuXVy1F7i9kf4RIpBAeiJktWWkRBZBUBStfwhRSE/6JyrK6EFCVEJgSoaFFWmkhNBDw9AI+mPDELEtt89vzjkzZ845d+5V6gu/nTnf3/t3Zs7cbTQ0InNjwyb7GvRiFBTfS2uj3rIX3wd9OxkQx+tbYx/Vav8vDJLSt6nfk4bVTNgmy7KL4zie4HrN2NjYKT3MLPeKgW9dl8zjnSWFzG61WrNc3kaNqoCxSZLkBmLehyyIk3gT67dsO3I15RruzUZAMTIycipTu5qgS5A5IZscmm+328MU8lWn0zkzaHACIO+WNE2flXtiL2I9TU0XWfoVyIulh0LtHhJglKAf4riH6xoCP8H9Ia6bsjQ7tzCMSl8ZBjZfpmk27gZEdwa675Ffkd903Jm2GdzzyH7kR2Qvsl14GTCDu0Du2a0LpUHq6FiuEdzXyEMW58DKROJHpIg4Tm5jOcPwJEpiFejnoaGh00oPBfh1kqhhtW3/FRB7JTZ/EGea+/FCoW0ofLHkpqFLCtICuqfx+8Dl4a5AN8V1nqw9RwMSPIPhUZq5rCCrzd+hJzhh83qHDiVJfJWsgwmiPP47xLidSU1j/41rmWbptUngcRPITqL7SHbe5k0EdOslvq2rAMelUnyvBAJ5x7TNjpzQ0Ql8P9x3lmmwSWx2cZFHaruOkw/E0q8l1k02p/n5yKvSXLfbbXc73bbSlFmyLJUB/NNsNs8uSAM5FFAeRI5IE67eYO7o6Ons0jF2YLIgZWeSZDOFvW+ZGlWBVL3X7+b3abpMN1g5EeG38TQM2ZzUI37y2JJjDJ+nsiwdtW2MncTE5saCtLb3QZ3wpUIZgExc2+11+J9ocp1Z+7sXSfH3IKvkXr5lsTpw/oI7TyxarfYs1jtdXwa6Vec0MpV/C90sUSMi5mEOuedchezAFu28oqqsQoqU9wf7jYbD/WRd6MqCCQCbt5H5Zs03bUJyMvHH87U6YF4oPQaHyhjJO7yb2jZUlDINij4qySjy8oqyUS0Xm425XZY+YLiM01UXuri09JvE5lt7LTsng0EO5jXw/iHX2zYe/LAKkVKx2+8RY6tpOIf+fh0TCR3/xo4GzhcbJnQgtU4yuIX6VFxouALaF/t58h5Vlbnvm/nA0vQWrp8FDwgXRZN2t+qeGK8gn7taUfBhjad7HjBRXuTrUgyP1nJbBZeJr7zcvQaM7yrs7i2ZoqArJSayQ8RS+ejFWyDGJzI0lxfF+rz4OLlZ1u6QKHAc/d80sdqoDEbnyskqu5A97OoM0G+QXyEuL4hVc7KL+c+xECqPXA2o/xdirXV5KWAOMol8UX1MItHdhUzR3N2BpyIH+gPIyyVTQv+8msyy6k88g1Q+/KrBpY6qf0vVODMT9ardGTTg8bwUo92JHPlxspr7J5EfkI9pbpHt4CZG/xp+n9ocfmfB7UOOKIkPc33MtpE4+gzYT4zZBVm9ce7DwH+BDCqzf4UFMEM+psitGC/BOHENQsnk/cP+d5mizQdrLRAkFRyVP1afoYblDHGXkOG8Nfn6ImqcRIN7kEdd1eDwC/CZRpDUT4H8B1J8xwNmzoQCFoYPqThdryPBn+YfUAVn0iHH44YfRAaL7JQnyNcK+lYwGC/vGPJGVeUMrgb1WhulJT/xmuTc1+maH+DBOAEqh8NHHuOBKa4ZHh4+x+Ur8MP6CObyGTnd5YB0+TB8/xPGfxiqURutUIVt/gX5diT/LqcYbgAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAZCAYAAADuWXTMAAABvklEQVR4XpVSPUvEQBBN8AvOShBOYrKTk6tiI0QQKxutxcY/YKeWYmGjv8A/YSeCCMKhVmJjpb0gKlYeVyloI55vbneTSbJBHLK7s/Pem9nZjed5vv084WjfaYV4JvtzzsydV0arjNpDOWXScfpmU3e8alzyhaeUWo7j+EGPllnNaMUPLbEHd13mGyxBEDRIqQsi1VekVtrt9liSJKPBdNCAaIqItjG+kWg1l4kUAN8wPtI0HRFgZsDuIJ7Tu0zv89ETgH1U7oiOPFRdsD4pugJvwqqM+Qxsshjgro2ilUn0fJmJifYsVjCQjnVlWkPPM4poEYmuMfY1Q7ZZMoi6EHxhvcEpbuG/cjIce0nynCn+KS688ywEfORzCzWbzXHse/xkRjfMd5BXNnrSb4j3pR2bkSHETi0Vr7CF/UZRqYETrhxG0bwQZzO/Ozj3/CPlx9YzV+hhvGM7pDFrmoO+D4Af2mIa8Af9pvxLAuzkmLYoCgPEjzB+wjBsZ2K+RQQf0eenEfNNv8B/wvqMeJdFpP/ns0JW2bOIlBxpBb5kVLM49V5dvJSrSqpGHKZJ1RacGY1lWGl2CzRQxeoiOfkX6NJky7oVCmIAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAAEp0lEQVR4Xp2XW2gdRRjHzza13m+YEJvkzOwmkcCJIhiqtUpBfJIiKBQfRMW2iiKiIiJKvRR8EKVaFH0RH2zVF0EQFcEL2IdSrbUPFS1SWiraFmuRoH2QWNv4+3Zmdr+d3T2J/mGyO//v/13muiedjkcSXgpopu29jv5WDadcfGQNUf4nzwUFEZLFubRpCr5NIPgfA2iULZisjddYjKYfGgfTp2hgjLl4bGzsbBH0Vzo0ahIXZ2Rk5JyqYZGYnJw8M8uy1dbaNbTlsT0eWLfbHSHhrvHx8QuVpUTbGBoGSZyeNXbH8PDwuQW5ECh2Kk3TD601+3huIsjzFP4nz0/TLBsOOl2HDBLN12mWritIXxAxLsL2I8/9aZrtd0//noV319C9FCLLu9TB65IiZjMSSfIkDr9S5F0QS0NxDAba7qYdGhoaOk+55GCQm5mh3Y6JlgL0er1l+D6NZp7nc+Q5a2Zm5gzZVqOjo5fIqhprjxpjtwQvJuMC9Meo5d56TNVHsIUAc1mWXl2yJUh2N/Z5a8yzgSsSsCK0VUpeIKTA/z2Km+e5UvPl4O3L2B7RJPr11PVzb7q3zKu1OQ96S16Uta9XBAqyh73mW+k730QG/DDc99VZcdAMmmO0E53qSl4psy9KbBsp9Nag95oB+L9t4HVAOUwYjtD+6o51R+rpHaamps5nS5xGN1uQSV7Q5yzxB6UyipD4w+YG/YnWUOhOOR/yju12BpEGt6BJ3d5/reS9DfJRCmIL2Dc0HwPdKp/8YMQfoG3WXAzsD3jfF5lldpuZoaBX6H8ZazWkDBksbW/JeEB+4YPeU7KCavUkus/rPlJ0vpTY7o/1gsDI/hZfnt8wQTt53yd9U5yXpOquuuheZWWOVwRysjHM+aDXxM4aUrDXFQdIZs8VYG+sqivuCZrfaH/wPqAK2o7/aqfI+9G3wikZ3EPYTnWKazEp7l/Zt6eHBsM1V585nKe97nA4TAL6V0nh8oxcHJL8UF+ea4z9uCCBrIJck9LlWryC/lblWYDcd0ruadHq2iAPSmC5NXKisLkX+cvHYqtoJIh2hpMpF35tQUbA/qDc3yz3Y7EtAM27tGs1F7IQ+xlsR7QtB0G3+eS3xTYBRa/D/o8sWbwa7qaR7ZM9UbJJZfDEf99PzApvLWQCbDfQ9gRamXJgoz6zQ1GdXIFhOW2W9pV8xfRM47AB/qT7enkyAprDaN7UXJD5L+bv1t3fA1ozyBdYBmzdGdug80arKod5W0GUSDrc3yuyNP0OwQGZWfbbC6n7/fAZ/etjD4Fayrfw265J/zX9wU/IPO0U7RfaIdpPNv9umJOeP5r/omyBlYNtzON58IaJEyyV+5V2B+I1GTdGLMgRORN0LUUcd4e2AfHeqEHzVc3ExESXWuaIfWnFUKIetM60Yol19/LGKh3NkB5AFLwtl3GH9h15r2hcJ/rbmKwObWKVbibBCZZ8VNHNaCy6noiY1xFzlpiXOaY+4Ar0AOpB2z1J8hTt7ZiPXdojVCCrKD+jb4oNDm1BGwdXZ2KwFzfxL9dgzAc0RoiSSpc4Kyl6vVK1oH/EqFvnHVPn9XtjCo0FBSX+BVFyCqMRUm+pAAAAAElFTkSuQmCC>