# **DORA Architecture & Technical Structure 🏗️**

DORAのアーキテクチャは、生物学的な階層構造と、コンピュータシステムのレイヤー構造を融合させた設計になっています。

## **🧠 Architectural Layers**

システムは大きく分けて以下の3層で構成されています。

### **Layer 1: The Hardware Abstraction (Micro-Level)**

* **実体:** snn\_research/hardware/event\_driven\_simulator.py (DORA Kernel)  
* **役割:** 物理的な神経活動のシミュレーション。  
* **データ構造:**  
  * **NeuronNode:** 膜電位 (![][image1])、閾値 (![][image2])、不応期状態を持つオブジェクト。  
  * **Synapse:** 接続先ID、重み (![][image3])、遅延 (![][image4]) を持つ軽量オブジェクト。隣接リスト形式で保持。  
  * **Event Queue:** (timestamp, neuron\_id) を格納する優先度付きキュー（Heapq）。  
* **動作原理:**  
  1. キューから最小時間のイベントを取り出す。  
  2. 対象ニューロンの膜電位を更新（Integrate）。  
  3. 閾値を超えたら発火（Fire）。  
  4. 接続先ニューロンへのイベントを、遅延を加えてキューに積む。  
  5. 行列演算は一切行わない。

### **Layer 2: The Middleware / Core (Meso-Level)**

* **実体:** snn\_research/core/snn\_core.py  
* **役割:** 抽象的なモデル定義と物理カーネルの橋渡し。  
* **Compiler:** PyTorchの nn.Module (Linear層など) を解析し、DORA Kernelのグラフ構造（ニューロンとシナプス）に変換します。この際、閾値以下の弱い結合は刈り取られ（Pruning）、スパース化されます。  
* **Interface:** forward\_step() メソッドにより、外部からのテンソルデータをスパイクイベントに変換してカーネルに投入し、結果を再びテンソルとして回収します。

### **Layer 3: The Cognitive OS (Macro-Level)**

* **実体:** snn\_research/core/neuromorphic\_os.py & cognitive\_architecture/  
* **役割:** システム全体の統合制御と認知機能の実現。  
* **機能:**  
  * **Prefrontal Cortex (PFC):** 意識（Global Workspace）の内容に基づき、トップダウンの指令を出す。  
  * **Basal Ganglia:** 競合するアクション候補から、現在の動機（Drives）に基づいて一つを選択する（Gating）。  
  * **Sleep Cycle:** システムをシャットダウンし、短期記憶（Hippocampus）を長期記憶に転送・整理する。

## **🔄 The Cognitive Cycle (Data Flow)**

DORAにおける情報の流れは、クロック同期ではなく、イベントの連鎖として表現されます。

1. **Sensation (感覚):**  
   * 外部入力（画像、テキスト）が SpikeEncoder によってスパイク列に変換される。  
   * OSがこれをKernelのイベントキューに push する。  
2. **Perception (知覚):**  
   * Kernel内でスパイクが伝播。  
   * VisualCortex などのモジュールが特徴を抽出。  
   * 予測誤差（Surprise）が発生した場合、可塑性（Plasticity）によりシナプスが即座に強化される。  
3. **Consciousness (意識):**  
   * 各モジュールの出力のうち、顕著性（Salience）の高いものが GlobalWorkspace にアップロードされる。  
   * ここで情報が統合され、"Conscious Broadcast" として全モジュールに共有される。  
4. **Action (行動):**  
   * PFC と Basal Ganglia が状況を評価し、MotorCortex に指令を送る。  
   * 物理的なアクション（またはログ出力）が実行される。

## **🧬 Key Mechanisms**

### **E/I Balance (興奮と抑制のバランス)**

DORAKernel は **Dale's Law** を実装しています。

* **興奮性ニューロン (80%):** 正の重みを出力し、他を発火させる。  
* **抑制性ニューロン (20%):** 負の重みを出力し、他の発火を抑える。  
  これにより、ネットワークは発火しすぎず、死滅もしない「クリティカルな状態」を維持します。

### **Synchronous & Asynchronous Execution**

* **Synchronous Mode:** デモやデバッグ用。入力を入れたら、その処理が終わるまで待機し、結果を即座に返す。  
* **Asynchronous Mode:** 本番OS用。入力はキューに積まれ、バックグラウンドの run\_loop がリソースの許す範囲で順次処理する。これにより、リアルタイム性と並列性が確保される。

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAXCAYAAAAC9s/ZAAABeklEQVR4Xn1RsUoEMRDdPSy0sj7YnclupYXYCJaWfoXloXCfoSD4B/6BIFx5lWAhNuKBYOkJJ/gDliKob0w2O9kkPgg7efPmzWS2KBzKfJC8WiTIBPUH7Rv00AU2toqY9ynPOD64RKIYOd7hfyNPBlTBzE9MvCLmF8RLxEtjzIFWg7/GecdZ4bwR0yR4QmPMMTP9IDnzpEZZrMH0wZjmEprtfjo3FRL7SIjBPDV2Xdd7yN0EvAaSY2tAz4n0CPw9JtjyjGusUUL0CZMPn3AicFOcU6VVUE4QvcoUbdtuquIxlrpA9/VOqRsEEf7CrRhUVbXTZbDtKxQfdp0Sxb2vjCkGWOiJdIThQiYICj0cqRNMNBEDIjrH9ww7mfbFccvoIqPaP8GPKL4DNep1Oajx5DeRNfhqmmY3Erg4nsgBy9tA8Te2fqH53PAJC2yd6Mj+shjJAkEuEfP6CUMMWP/OZEGwhdyrAlEYB8GgQxwnRyg8nyzQ8PUDpbv+ArJ9P2vc5MS7AAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAYCAYAAAAPtVbGAAACoklEQVR4XpVUv2sUQRjdjRFPI1hoPC+7Nz/uIiSFWFxABCVlIIqVhY2NIAqx8A9QESIIYmEnWigWQiCQUkEECxFElCgWmhghgp2VPxBEML5vZ2Z3fu2ij5vbmTfv+7433+xdkvwH0mDiIXU1jkwtfNZFVPMveusRxDejWRnfbXAXD3DhuA5AfHSv7lghaXciJK055+w153ydcf4B8zXO+JoQYtqSQsMXMT5Ds47nJ8bY6WiBEhESSc8geANjKYkbHJZCvIDuFjSTFR0Xu9AbCD6gizyIqbvd7hRO+djnQ6hWqwxeHiTv6CJvK7ZUD4F/BiMT9laIUh9Z6rJI9Avjqx8Pbg5j3udd1O2Gp/lIp+n1ejuMC7SITvhKCtFy1Z5ljylBZLmdFkWeIOlG3s33GQXeogW0aSaWtBk1GiS7RyfBazqr1zNUxNcltr+kMHcK5lbw6p+zNBp+uxifpyJIflZI0aI2Ubv0QSsd5/fBz9okuPcYBytGwz8Q/cCoXXhexWmuYD4X6dIwkn2HoY7Zgqk9xA0Gg82VOPEK6AW1B8HUrpcIegpqyJZRWxjjd/H8Ce2NLMtyHXdC3Sc/gnENJg/ZcQUsRxOs+K2w31LK/Y4IIKdIcAmJbgv9tlEsuJvgllF0J/hpzN84gaqA+s7zfCsK/CE3tsYGTDwk5859MP4O/wiH1T47Cs3zareAezNwddK4DC6tuA/2DSY6huj3+7ux/mHuAwXuYH2xCnGgXqEwrwLx9P8lpFilORJdGMvGtsHUcZh6ZGTgv2BMSiEul4FRlBuuAu3M4HSF2omCx4jD/DqKnFeKNEXrltW9seJ1rjLYuWorqy0hZGt87/gWI2u32yPVboFNo6O7tpuFBy97zWlq0aRXlHcU9QlR56NhTvgLvsh5w+i8xQgAAAAASUVORK5CYII=>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAYCAYAAAAVibZIAAACNklEQVR4XoWSvYsVMRTF56mIH5U2D9/MJBmxepWgCCKWwpY2Kgr+AepWFoKCYCPaiYXNwsoiWK2VhbKInZ2FoCwI68eytViIaOXq72aSmSTz4eHdN7nn3Htyk5ksE0zsbxy9NV3GoRaG5IYfKsjimqis2zzkMsT3YLy0q3aZQSTzp8fxZPM/iEmmtf5AbBIbxBfiK1FNp9P9PNe1VpJvKDRjzPuiKPZKJ+srSqkt17uplX4cWSPeQ/hrKnPT7iRw48CviobpVdKd4b3Tdw7tE89jQkQnQFi0jUrdSI/GpI/shsZctERzLZxSadFOet6hrmCKS27S+40EqqqaYvpDNOJa2MkAc2KlrU58mWLBNS4Fp5MTLGG67I5/O5BEe5bneeG5DpjohDNd9RwbHSVfZprzVlPqgZ9FhiDuyDq8rujqeKNHnOlrz9H0nPwQpme4OzFdcdIuuLXZLN8nSY9p/cD0oDN9JzmGZ1nfknVZlsfFVDaRevhFTC87n1Hs4O62lVZb8/l8N+tXmOwRQSl92G34BrMDbPAyS05qUb/zNhHQ9J34SVzH8IKXrFFtug7/kPxU1Bigw9D0WY7JW35hifYzmKD9IX5xgqdNQ4SOXQ2a3mK6XfElWCKoQ/tG/GZSk0jj4O7WMH3SEMEdYfiRE9z1fBfRhbYoy+K0fAV1FheZyizks5n9hFr0uXgk2lhpulma/gdh9XhnLY3VNFr4Hy6StN8oae5lkrTfyKKVevtHOhOklf8ATT90oi8ULI8AAAAASUVORK5CYII=>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADQAAAAYCAYAAAC1Ft6mAAAE3klEQVR4XrWXa4hVVRTH7x2zdxTFdHXm3rPPmbk200BQ3iKJijKTgqD6lFGgXyoLg6QvvaD6JIO9oCCkl2FQVmhWEFSEWkQPQqjoS01mSSlFBqXphE6/dfbeZ9Y595xzzyX7w5qz11r/9diPs8+dWk1Qj//agR7nuhJLOTJ5eqIiLUF/fNt9fzEV0L06uShxWfQkHAXk1ijvO0G/m9mNVGCVLJpThV+CklR9Ze6L3Aeq5NWckZGRU4eGhk5UpmLkbp3d9lxXVm+1WhcYY75FfkH2ID8jO5Ep5Afk6yAI1kRR1FBhCsXLH4bhZGDM3+SYIcediUN3lm0uD1U4As2j4CtSmCYWedvg4ODJ6Kux/2QCMzU8PNysnN3RiF9J7Aw5zksTNEpylrhKQcHvkX3ED6Q9dfEtdpN9UdvLYL11Waj1xP5R68pbAL17/WE2QFZeVpGGt6QTJeM5NLUXmWZ8jCL0hDu2b2XtCXL7zjVWBwVvcjtwV9bngf/jwHJC0XXJZrO5AP/1URgZZZbjFkpe5G5t92Ahz2AHr+Q9Hs36BJ1OZy6xlzcajZOyvrGxsVNSBn8kBAQ94yZ0bjfHAv9vMScK53lbFEWXEvMl9g3IHYy3I8vEJ7GMV8jOsxALk0QOckkQ8x3yALIZ7jqeX3h/s9k6Hf0d5ClkNwmTkwF3UnMdZts19pb7vRafcz1Vi5AbTiaD7Pcuxlcj/wSBWeN5URQuZQKveZ2mX4Czj5DU+8OirMB+wJggkmwL2u3j0P+E/6nnoD/Jzp0fyqVia5/tfUF8EwcbUk16xO+PBATBG0Vnl0K3Cid0lwJH7AT0X8Mw+gx1QL4xFL8E2w75DMR56nFTO4l5U6cl1wj2aeRRb3P5pml0rethAP0RGRC/hfGumnOgj7tebrPRYk4XuNkR0u+P4uB/33HiK53nDXYRzDc8X2eX1mN7aHR0tOVjOI4mjonC1UkiO8n7xE7dK7yR2MuszVyTcGtxndOwHzTxRG1L2G53vYwnRL0POJ9zhNz3h8JL8B9BHhOjS3qv+7YsTpHVGM5yyWsy3x/0z5FD+pcDNR7EdlgmoLnoyzhakuMiZXsVfY/mpWDsr4Kc9yf+/sgy70be6yzszPV9hznn2oMjNyRPfM+b+PtTl+Oztt1uD7qcWzlan+gYJrQN+w4X9yyPOVJLfqGgH5LbznPR93LUN3rdwq00x2JMGgvlnCtwdM7EvkoawjdZcwU83HtwkOO03NukKPbHHV8Kvy3Nox/P88OYZI/c/ciU2km5IGaQlxjPkx3wOUP7K+Ww/GJx+o1xv1G40nM8cZGxV+Z+l+xAYIIfjf0I7hKB84TcMrNRbhVmc0hyuW3kyn8Z/V3kOuW/1vrNJhZuqbfLYsltJnY4G3mu4vkwz7+QzVwQ5/hCbXv7fYR8ENhXQ3qe4cfuWT5fCvqAaUuRmsXExMSxNDMuv6azPomVDyINzM/LwyRNqN4Z9IZMQHN8XiYzX945N7GvNKcEeno5HcQosv8X5FdkF2/h4jnCpC90+hL3S+Uq0VP8bPD/j6KKRfaaHNl1yHaGA0zmYmO/afeIL4nKD68XORQyq9iTX+viVAnRkKMayL80gdnE8+mIn1lZTgrlBYq8RfYMimhF9gR64XqSjwKq1CjiFNkTaEK9lJ91/QvRJyDsZDKB/QAAAABJRU5ErkJggg==>