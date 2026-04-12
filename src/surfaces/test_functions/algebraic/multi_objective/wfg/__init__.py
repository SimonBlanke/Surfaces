# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""WFG multi-objective test function family.

The WFG (Walking Fish Group) problems provide a comprehensive toolkit
for multi-objective optimization benchmarking with configurable
transformation pipelines.
"""

from .wfg1 import WFG1
from .wfg2 import WFG2
from .wfg3 import WFG3
from .wfg4 import WFG4
from .wfg5 import WFG5
from .wfg6 import WFG6
from .wfg7 import WFG7
from .wfg8 import WFG8
from .wfg9 import WFG9

__all__ = ["WFG1", "WFG2", "WFG3", "WFG4", "WFG5", "WFG6", "WFG7", "WFG8", "WFG9"]

wfg_functions = [WFG1, WFG2, WFG3, WFG4, WFG5, WFG6, WFG7, WFG8, WFG9]
