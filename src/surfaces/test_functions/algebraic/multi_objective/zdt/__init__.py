# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""ZDT multi-objective test function family.

The ZDT problems (Zitzler, Deb, Thiele) are widely used benchmarks for
bi-objective optimization with different Pareto front geometries.
"""

from .zdt1 import ZDT1
from .zdt2 import ZDT2
from .zdt3 import ZDT3
from .zdt4 import ZDT4
from .zdt6 import ZDT6

__all__ = ["ZDT1", "ZDT2", "ZDT3", "ZDT4", "ZDT6"]

zdt_functions = [ZDT1, ZDT2, ZDT3, ZDT4, ZDT6]
