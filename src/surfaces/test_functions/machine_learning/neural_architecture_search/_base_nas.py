# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BaseNeuralArchitectureSearch(MachineLearningFunction):
    """Base class for Neural Architecture Search test functions."""

    _spec = {
        "category": "neural_architecture_search",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
        "evaluation_cost": "very_high",
        "initialization_cost": "very_high",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
