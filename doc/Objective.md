# **Project Objective: Why DORA? 🎯**

## **1\. The Problem with Current AI**

現在の人工知能（Deep Learning / Transformer）は、確かに強力ですが、生物学的知能とは決定的に異なる「工学的な嘘」の上に成り立っています。

* **静的である:** 学習（Training）と推論（Inference）が完全に分離されている。世界は常に変化しているのに、AIの重みは固定されている。  
* **効率が悪い:** 脳が20Wで動作するのに対し、同等のLLMは数kWを消費する。これは、「何も起きていない場所でも計算している（密行列演算）」ためである。  
* **時間がない:** RNN以外、時間の概念が空間（シーケンス長）に置き換えられている。「タイミング」による情報処理が行われていない。

## **2\. The Solution: DORA's Approach**

DORA (Distributed Online Reactive Architecture) は、これらの問題を解決するために、脳の物理的な制約をあえて導入します。

### **「計算しない」ことによる高速化**

行列演算 ![][image1] は、入力 ![][image2] の大半がゼロであっても、全ての要素に対して掛け算と足し算を行います。

DORAは、**「スパイク（変化）が届いたニューロンだけが更新される」** イベント駆動型アーキテクチャを採用します。

脳の発火率は非常に低い（スパースである）ため、これにより計算量を劇的に削減できます。

### **「予測」による学習**

DORAは、外部からの教師あり学習（Backpropagation）を行いません。

代わりに、**「自らの予測と、実際の入力とのズレ（Surprise）」** を最小化するように、局所的にシナプスを変形させます。

これは、生物が環境に適応するメカニズムそのものです。

### **「OS」としての脳**

我々は、SNNを単なるパターン認識器ではなく、**コンピューティングシステムそのもの（Operating System）** として捉え直します。

* ニューロン \= プロセス / スレッド  
* スパイク \= メッセージパッシング / 割り込み  
* エネルギー \= 計算リソース（CPU時間、電力）  
* 睡眠 \= メモリのデフラグ、ガベージコレクション

## **3\. Ultimate Goal**

DORAの最終目標は、特定のタスク（画像認識や翻訳）を解くことではありません。

**「自律的に環境と相互作用し、エネルギーを維持しながら、経験から学習し続ける人工生命体」** を、一般的なコンピュータ（CPU）上で動作させることです。

*"We are building a brain, not a matrix multiplier."*

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC4AAAAYCAYAAACFms+HAAADlklEQVR4Xr1UTYhNYRg+x8wwWAhxc8+d75zvzpW6LOSKJFlIzYLyk59RNqwMKwtFkY2/jd+URjREytDIgiQWsmKhTFMKg1nLQmIhM573nO+c853v59xzR3mad873Pe/z/nzvuedznH+FqxIpEpdNI/OuSWYkm6DlgCyKhEeaIkonX6YMIAfyJHKUTRO21HqEzEvIbJrB1k3Bwzh2r3wMm6Y5bP0lMJIRbC7bweQz5yCjoY3vs2Gf+Z9933+P9Sien2C8VCrNxHOEhXsGnz8aBMHbSqUynWKx3seYPyZiya5nM+f2OjmogYyxUyg8gWYOR+5UAX6QfLA+bNsShxPGbQP/Ac8GQkSQmr0AbFOXSZMAxQ9gchNo4FDW4zrgL0eH4r1qMHjyrcqy/xEovktM/LTMc85L4L+Lie+P2Kh7HLIOG0jV+tiMAyuiUWDj6ffaI5rrl3n8vvvBXSMf1kdjnhKBu+eVvYokT2GtlOuK4Ib9LMVQ1tUW1qYRVavV5oFbmQhicB6sEM0NxhwFU9NIsD30MXYu9fGegAfH4711nAURRzUajQ7UvQu7gZoXYcOoewzPq7Ah2FmphOvgpqixaOLPYhbBD7FfgMD1ovGBKMZtx/qJVy7PSDO0BpseNc/AdtK6Xq9PpboY0AvP8+biZvuN/fNUjSxofA6JYG9Egk1YH6ECXV1dy+nD5TgI+fzwQ2a7tV9n9GcE8vXCXuHAi1VfCBEYvlWRh7Sip72kwPMg53yREuNOgWMcDY2Jkz5FoU7yIkFVJHiJ9Wy8mcdJmAWJUzSB2Dsixx5JlgoMW2j7ohjGMwIVEH2D/aCToekdMU/NIngCUx8BfwH71Wox41rahm+UsQ2NxrKOlNfFMoMB3UcvX1LG0DQBzX0UU3kUc0Lqouk/4H/CbtvizVC6SdbG1unbOYkaW7BuE4O8FTvRw1bhywLka9g4xw1j8H2F/Qp4EKi+VqCfOWVwU60Vg+vDATbSd0UfK2mq1eos8ENYt2tJ6KaA82aGFCLw72AnVF6DMlEdNt5xuru759NNBjuPWpdgm2Gf0dcV2ANcEkvUmDAdHGvot5gwUg1MuqccX3/22hkUlGnApHm0cuk26kRPntgKTDZzCD3YmFeXORopB2bnZUYqaCp17JoilXToITl5bOdsGTmROa4E8oCLQxXbsiQDUAOyyPfKyFfKXpvSxhugHyZnGyE5sAIjqUCJlQcn/yf8BfU1rlVxW+ikAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAsAAAAYCAYAAAAs7gcTAAABWElEQVR4XnVQO07EMBD1iuXTUSAoyGbGzgYhBQqkFJRbUFOwNPR0W3IASiQaQNAgUSFRISEQB6DgClwALsAFKJbnJGsmEzOS7Tdv3ht7bIzpGR/1XoOAA98wUhRA1NBK/hPp3lGRAh5H3B2nTHWt3uMGWYgrJBX2wEizNshQNsUIt7Vuh5j28jxf9GS+ka86a3dbN5RlOW+dfbTW3jPzNRN/ENEpM90BP4O7CI0hOsc68mlRFAsQTZG/ryfJCoQ/WG9NX2PQ5XLmBN5CcQrDsSeI+cQ5uxnEMiCa4GqI2ela56cgeoLhS0rEh/T6+IEziMbAczi/sR5mIuBDmMdVjkFG1RuJJ3jvvsd+YF/LsmzZ/wZgvxIPs+Ea/vIVgisIbzDQAc5P3HbLRC9pmm7XLxAhB4J5aTBIE4/lPH+hBpU1MZ9gQ9olG0YVQqir4qF6t0BooHYpatJfyWMyjf2I8jgAAAAASUVORK5CYII=>