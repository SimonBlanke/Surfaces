# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate model loader for fast function evaluation.

The training infrastructure, validation tools, and dashboard have moved
to the surfaces-surrogates package (pip install surfaces-surrogates).
"""

from ._surrogate_loader import (
    SurrogateLoader,
    compute_interface_fingerprint,
    get_surrogate_path,
    load_surrogate,
)

__all__ = [
    "SurrogateLoader",
    "compute_interface_fingerprint",
    "load_surrogate",
    "get_surrogate_path",
]
