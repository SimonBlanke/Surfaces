# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_machine_learning import MachineLearningFunction


class BaseLLMOptimization(MachineLearningFunction):
    """Base class for LLM Optimization test functions."""

    _spec = {
        "category": "llm_optimization",
        "continuous": False,
        "differentiable": False,
        "stochastic": True,
        "evaluation_cost": "medium",  # Mock mode is fast
    }

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
