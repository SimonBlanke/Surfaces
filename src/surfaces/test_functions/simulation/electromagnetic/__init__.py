# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Electromagnetic simulation functions.

This module provides optimization benchmarks for electromagnetic systems:

ODE-based (scipy.integrate):
- Electrical circuits (RLC, RC filters)

FDTD-based (future, requires meep):
- Waveguide optimization
- Antenna design
- Photonic device optimization

Examples
--------
>>> from surfaces.test_functions.simulation.electromagnetic import RLCCircuitFunction
>>> func = RLCCircuitFunction(target_frequency=100.0)
>>> result = func({"R": 10.0, "L": 0.01, "C": 0.0001})
"""

from .circuits import RCFilterFunction, RLCCircuitFunction

__all__ = [
    "RLCCircuitFunction",
    "RCFilterFunction",
]

electromagnetic_functions = [
    RLCCircuitFunction,
    RCFilterFunction,
]
