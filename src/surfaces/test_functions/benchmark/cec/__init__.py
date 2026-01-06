# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""CEC Competition Benchmark Functions.

This module provides test functions from the IEEE Congress on Evolutionary
Computation (CEC) benchmark suites. These functions are widely used in the
optimization research community for comparing algorithm performance.

Available suites:
- CEC 2013: 28 functions (unimodal, multimodal, composition)
- CEC 2014: 30 functions (unimodal, multimodal, hybrid, composition)
- CEC 2017: 30 functions (simple, hybrid, composition)
"""

from . import cec2013, cec2014, cec2017
from ._base_cec import CECFunction
from .cec2013 import *
from .cec2014 import *
from .cec2017 import *
