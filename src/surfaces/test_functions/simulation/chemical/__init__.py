# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Chemical kinetics simulation functions (Cantera-based).

This module provides combustion and reaction kinetics benchmarks including:
- Ignition delay optimization
- Emission minimization (NOx, CO)
- Reactor design optimization

Requirements
------------
- cantera
- numpy

Examples
--------
>>> from surfaces.test_functions.simulation.chemical import IgnitionDelay
>>> func = IgnitionDelay(fuel="methane", mechanism="gri30")
>>> result = func({"temperature": 1200, "pressure": 10, "phi": 1.0})
"""

__all__: list = []
chemical_functions: list = []
