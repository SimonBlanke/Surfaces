# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BaseEnsembleOptimization(MachineLearningFunction):
    """Base class for ensemble optimization test functions."""

    _spec = {
        "category": "ensemble_optimization",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
