# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BaseDataAugmentation(MachineLearningFunction):
    """Base class for Data Augmentation test functions."""

    _spec = {
        "category": "data_augmentation",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
        "evaluation_cost": "very_high",
        "initialization_cost": "very_high",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
