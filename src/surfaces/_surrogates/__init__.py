# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model support for fast function evaluation.

This module provides:
- SurrogateLoader: Load and run pre-trained ONNX surrogate models
- SurrogateTrainer: Train new surrogate models (for maintainers)
"""

from ._surrogate_loader import (
    SurrogateLoader,
    load_surrogate,
    get_surrogate_path,
)
from ._surrogate_trainer import (
    SurrogateTrainer,
    train_surrogate_for_function,
)

__all__ = [
    "SurrogateLoader",
    "SurrogateTrainer",
    "load_surrogate",
    "get_surrogate_path",
    "train_surrogate_for_function",
]
