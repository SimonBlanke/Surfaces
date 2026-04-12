# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Discrete (pseudo-boolean/combinatorial) test functions."""

from ._base_discrete_function import DiscreteFunction
from .knapsack import KnapsackFunction
from .leading_ones import LeadingOnesFunction
from .nk_landscape import NKLandscapeFunction
from .onemax import OneMaxFunction
from .trap import TrapFunction

__all__ = [
    "DiscreteFunction",
    "KnapsackFunction",
    "LeadingOnesFunction",
    "NKLandscapeFunction",
    "OneMaxFunction",
    "TrapFunction",
]

discrete_functions = [
    OneMaxFunction,
    LeadingOnesFunction,
    NKLandscapeFunction,
    TrapFunction,
    KnapsackFunction,
]
