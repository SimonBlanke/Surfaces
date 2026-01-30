# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .tabular import (
    StackingEnsembleFunction,
    VotingEnsembleFunction,
    WeightedAveragingFunction,
)

__all__ = [
    "VotingEnsembleFunction",
    "StackingEnsembleFunction",
    "WeightedAveragingFunction",
]
