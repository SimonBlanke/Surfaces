# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BasePipeline(MachineLearningFunction):
    """Base class for pipeline optimization test functions."""

    _spec = {
        "category": "pipeline_optimization",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
