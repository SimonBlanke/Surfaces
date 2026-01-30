# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .test_functions import (
    StackingEnsembleFunction,
    VotingEnsembleFunction,
    WeightedAveragingFunction,
)

__all__ = [
    "VotingEnsembleFunction",
    "StackingEnsembleFunction",
    "WeightedAveragingFunction",
]
