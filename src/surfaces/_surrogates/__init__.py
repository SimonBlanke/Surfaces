# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model support for fast function evaluation.

This module provides:
- SurrogateLoader: Load and run pre-trained ONNX surrogate models
- SurrogateTrainer: Train new surrogate models (for maintainers)
- SurrogateValidator: Validate surrogate accuracy against real function
"""

from ._surrogate_loader import (
    SurrogateLoader,
    get_surrogate_path,
    load_surrogate,
)
from ._surrogate_trainer import (
    SurrogateTrainer,
    train_surrogate_for_function,
)
from ._surrogate_validator import (
    SurrogateValidator,
)

__all__ = [
    "SurrogateLoader",
    "SurrogateTrainer",
    "SurrogateValidator",
    "load_surrogate",
    "get_surrogate_path",
    "train_surrogate_for_function",
]
