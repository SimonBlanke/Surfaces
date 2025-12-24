# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model support for fast function evaluation.

This module provides:
- SurrogateLoader: Load and run pre-trained ONNX surrogate models
- SurrogateTrainer: Train new surrogate models (for maintainers)
- SurrogateValidator: Validate surrogate accuracy against real function

Developer API for ML surrogates:
- train_ml_surrogate: Train surrogate for single ML function
- train_all_ml_surrogates: Train all registered ML surrogates
- train_missing_ml_surrogates: Train only missing surrogates
- list_ml_surrogates: List registered functions and status
"""

from ._ml_surrogate_trainer import (
    MLSurrogateTrainer,
    list_ml_surrogates,
    train_all_ml_surrogates,
    train_missing_ml_surrogates,
    train_ml_surrogate,
)
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
    # Loader
    "SurrogateLoader",
    "load_surrogate",
    "get_surrogate_path",
    # Generic trainer
    "SurrogateTrainer",
    "train_surrogate_for_function",
    # Validator
    "SurrogateValidator",
    # ML-specific trainer (developer API)
    "MLSurrogateTrainer",
    "train_ml_surrogate",
    "train_all_ml_surrogates",
    "train_missing_ml_surrogates",
    "list_ml_surrogates",
]
