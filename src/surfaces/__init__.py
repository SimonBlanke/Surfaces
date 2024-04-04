# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("surfaces")
__license__ = "MIT"

from .mathematical_functions import (
    mathematical_functions,
    mathematical_functions_1d,
    mathematical_functions_2d,
    mathematical_functions_nd,
)

from .machine_learning_functions import machine_learning_functions

test_functions: list = mathematical_functions + machine_learning_functions


__all__ = [
    "test_functions",
    "mathematical_functions",
    "mathematical_functions_1d",
    "mathematical_functions_2d",
    "mathematical_functions_nd",
    "machine_learning_functions",
]
