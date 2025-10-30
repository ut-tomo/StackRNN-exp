# Stack-RNN 再現実験

## 概要

このプロジェクトは, Stack-Augmented Recurrent Neural Networks (Stack-RNN) を用いた形式言語学習の実験を行うためのリポジトリである.

### 目的

1. **Stack-RNNの再現**: [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007) (NeurIPS 2015) で提案されたモデルと実験の再現
2. **ベースラインモデルとの比較**: Transformer, Mamba などの現代的なアーキテクチャとの性能比較
3. **拡張実験**: Neural Turing Machine (NTM) の一種である [Learning to Transduce with Unbounded Memory](https://arxiv.org/abs/1506.02516) の実装と評価

## プロジェクト構成

```
StackRNN-exp/
├── src/
│   ├── models/
│   │   ├── stack_rnn.py           # Stack-RNN実装
│   │   └── baselines/
│   │       ├── lstm_model.py       # LSTMベースライン
│   │       ├── transformer_model.py # Transformerベースライン
│   │       └── mamba_model.py      # Mambaベースライン
│   ├── data/
│   │   └── tasks.py                # 形式言語タスク生成
│   ├── training/
│   │   └── train_baseline.py      # ベースラインモデル学習スクリプト
│   ├── func_test/
│   │   ├── test_lstm.py            # LSTMテスト
│   │   ├── test_transformer.py    # Transformerテスト
│   │   └── test_mamba.py          # Mambaテスト
│   └── config/                     # 設定ファイル
├── cpp_ref/                        # C++参照実装
└── README.md
```

## 実装済みタスク

形式言語学習タスク（`src/data/tasks.py`）：

1. **Task 1**: $a^n b^n c^n ...$ - カウンティングタスク（文字数可変）
2. **Task 2**: $a^n b^{kn}$ - 繰り返しパターン（$k$は指定可能）
3. **Task 3**: $a^n b^m c^{n+m}$ - 加算タスク
4. **Task 4**: Memorization - 文字列の記憶と逆順再生
5. **Task 5**: $a^n b^k c^{nk}$ - 乗算タスク
6. **Task 6**: $a^n b^m c^n d^m$ - 交差構造パターン
7. **Addition**: Binary Addition

## ベースラインモデル

以下の3つのベースラインモデルを実装：

- **LSTM**: 標準的なLong Short-Term Memory
- **Transformer**: 自己注意機構ベースのモデル（因果的マスク付き）
- **Mamba**: State Space Model (SSM) ベースの効率的なモデル




## 今後の予定

- [ ] Stack-RNN本体の実装完成
- [ ] Neural Turing Machine (NTM) の実装再現
- [ ] 逆ポーランド記法タスク
- [ ] 括弧マッチングタスク

## 参考文献

1. Joulin, A., & Mikolov, T. (2015). [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](https://arxiv.org/abs/1503.01007). NeurIPS 2015.
2. Grefenstette, E., Hermann, K. M., Suleyman, M., & Blunsom, P. (2015). [Learning to Transduce with Unbounded Memory](https://arxiv.org/abs/1506.02516). NeurIPS 2015.
