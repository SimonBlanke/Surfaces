# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from typing import Any

from .._base_ensemble_optimization import BaseEnsembleOptimization


class BaseTabularEnsemble(BaseEnsembleOptimization):
    """Base class for tabular ensemble optimization test functions."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
