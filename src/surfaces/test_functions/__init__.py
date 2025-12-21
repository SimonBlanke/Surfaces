# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

from .algebraic import algebraic_functions
from .algebraic import *  # noqa: F401,F403

from .machine_learning import machine_learning_functions
from .machine_learning import *  # noqa: F401,F403

from .engineering import *  # noqa: F401,F403

# Engineering function list
engineering_functions = [
    "ThreeBarTrussFunction",
    "WeldedBeamFunction",
    "PressureVesselFunction",
    "TensionCompressionSpringFunction",
    "CantileverBeamFunction",
]

# Backwards compatibility alias
mathematical_functions = algebraic_functions

test_functions: list = algebraic_functions + machine_learning_functions + engineering_functions
