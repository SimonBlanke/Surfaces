# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import importlib.metadata

__version__ = importlib.metadata.version("surfaces")
__license__ = "MIT"

# Expose presets and modifiers at package level
from . import modifiers, presets
