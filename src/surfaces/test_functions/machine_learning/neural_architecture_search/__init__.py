# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .cnn import CNNKerasNASFunction
from .mlp import MLPPyTorchNASFunction

__all__ = [
    "MLPPyTorchNASFunction",
    "CNNKerasNASFunction",
]
