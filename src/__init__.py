"""
StackRNN-exp: Stack-augmented RNN実装と比較実験

このパッケージは、Stack-RNNの実装とLSTM、Transformerなどの
ベースラインモデルを提供します。
"""

__version__ = "0.1.0"
__author__ = "ut-tomo"

# 主要なモデルを公開
from .models import LSTMModel, TransformerModel

__all__ = [
    'LSTMModel',
    'TransformerModel',
]
