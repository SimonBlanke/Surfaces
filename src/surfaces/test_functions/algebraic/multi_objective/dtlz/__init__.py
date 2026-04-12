# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""DTLZ multi-objective test function family.

The DTLZ problems (Deb, Thiele, Laumanns, Zitzler) are scalable
benchmarks where both the number of objectives and dimensions can
be configured.
"""

from .dtlz1 import DTLZ1
from .dtlz2 import DTLZ2
from .dtlz3 import DTLZ3
from .dtlz4 import DTLZ4
from .dtlz5 import DTLZ5
from .dtlz6 import DTLZ6
from .dtlz7 import DTLZ7

__all__ = ["DTLZ1", "DTLZ2", "DTLZ3", "DTLZ4", "DTLZ5", "DTLZ6", "DTLZ7"]

dtlz_functions = [DTLZ1, DTLZ2, DTLZ3, DTLZ4, DTLZ5, DTLZ6, DTLZ7]
