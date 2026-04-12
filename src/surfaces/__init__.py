# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("surfaces")
__license__ = "MIT"

from .collection import collection
from .test_functions._custom_test_function import CustomTestFunction

__all__ = ["collection", "CustomTestFunction", "__version__", "__license__"]
