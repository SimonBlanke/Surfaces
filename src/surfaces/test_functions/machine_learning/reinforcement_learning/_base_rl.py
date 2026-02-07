# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BaseReinforcementLearning(MachineLearningFunction):
    """Base class for Reinforcement Learning test functions."""

    _spec = {
        "category": "reinforcement_learning",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
        "evaluation_cost": "high",
        "initialization_cost": "high",
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
