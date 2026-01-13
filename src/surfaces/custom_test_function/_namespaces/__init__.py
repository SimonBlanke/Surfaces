# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Namespace classes for CustomTestFunction."""

from ._analysis import AnalysisNamespace
from ._plot import PlotNamespace
from ._storage import StorageNamespace
from ._surrogate import SurrogateNamespace

__all__ = [
    "AnalysisNamespace",
    "PlotNamespace",
    "StorageNamespace",
    "SurrogateNamespace",
]
