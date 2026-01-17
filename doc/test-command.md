# **Neuromorphic Research OS: 実験・テストコマンドガイド (v6.0)**

このドキュメントでは、**Neuromorphic Research OS (NROS)** 上での実験、観測、およびシステム検証を行うためのコマンドをまとめています。

本プロジェクトは「タスクを解くAI」ではなく、\*\*「知能現象（覚醒・睡眠・可塑性）を観測するOS」\*\*へと移行しました。

したがって、従来の単発デモよりも、**ライフサイクルを通じた観測実験**が推奨されます。

## **⚠️ 実行環境について**

すべてのコマンドは、プロジェクトのルートディレクトリで実行してください。

モジュールパスの問題を避けるため、各スクリプト内では自動的にパス調整を行っていますが、エラーが出る場合は以下のように実行してください。

\# Mac/Linux  
export PYTHONPATH=.  
python scripts/...

\# Windows  
$env:PYTHONPATH="."  
python scripts/...

## **1\. 標準観測実験 (Core Experiments)**

Neuromorphic OS v6.0 カーネルを使用した、推奨される標準実験です。

### **A. 覚醒・睡眠サイクル実験 (Basic Life Cycle)**

脳のエネルギー代謝（Astrocyte）と意識レベル（Global Workspace）の相互作用により、自律的に覚醒と睡眠を繰り返す様子を観測します。

* **実行コマンド:**  
  python scripts/experiments/run\_research\_cycle.py

* **観測項目:** エネルギー準位、疲労度、意識レベルの自発的な振動。  
* **出力:** runtime\_state/experiment\_history.json

### **B. 記憶定着と構造的可塑性 (Memory & Plasticity)**

学習（Wake）→ 睡眠（Sleep）→ 再学習（Wake）のサイクルを実行します。

睡眠中の「海馬リプレイ（夢）」と「シナプス刈り込み（Pruning）/生成（Genesis）」による脳構造の変化を観測します。

* **実行コマンド:**  
  python scripts/experiments/learning/run\_memory\_consolidation.py

* **観測項目:** 認識精度(Accuracy)、ドーパミン報酬、有効シナプス数の物理的増減。  
* **出力:** runtime\_state/memory\_experiment\_history.json

## **2\. データの可視化 (Visualization)**

実験で生成されたJSONログを解析し、グラフとして出力します。

* **基本サイクルの可視化:**  
  (run\_research\_cycle.py の結果を表示)  
  python scripts/visualization/plot\_research\_data.py

  → 出力: experiment\_result.png  
* **学習と脳構造変化の可視化:**  
  (run\_memory\_consolidation.py の結果を表示)  
  python scripts/visualization/plot\_memory\_learning.py

  → 出力: memory\_learning\_result.png

## **3\. システムヘルスチェック (System Verification)**

OSカーネルや各モジュールが正常に動作しているかを確認します。

* **プロジェクト健全性チェック (推奨)**:  
  ディレクトリ構造や必須ファイルの存在、基本的なインポート確認を行います。  
  python scripts/tests/run\_project\_health\_check.py

* **全ユニットテスト実行**:  
  python scripts/tests/run\_all\_tests.py  
  \# または  
  python \-m pytest tests/

## **4\. 従来機能・個別モジュール実験 (Legacy & Components)**

旧バージョン(v17.x以前)のデモや、特定機能単体の検証スクリプトです。

これらは新しいOSカーネル上ではなく、個別のSNNモデルとして動作する場合があります。

### **Brain Components (脳機能モジュール)**

* **視覚野 (Forward-Forward則)**: python scripts/demos/visual/run\_spiking\_ff\_demo.py  
* **自由意志・意思決定**: python scripts/demos/brain/run\_free\_will\_demo.py  
* **クオリア・内部表現**: python scripts/demos/brain/run\_qualia\_demo.py

### **Social & Systems (社会・システム)**

* **集合知**: python scripts/experiments/systems/run\_collective\_intelligence.py  
* **エージェント実行**: python scripts/agents/run\_autonomous\_learning.py

### **Benchmarks**

* **レイテンシ測定**: python scripts/benchmarks/benchmark\_latency.py

## **5\. デバッグ・診断**

* **スパイク活動のモニタリング**:  
  python scripts/debug/debug\_spike\_activity.py

* **シグナル診断**:  
  python scripts/debug/diagnose\_signal.py  
