# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Chemical simulation functions.

This module provides optimization benchmarks based on chemical systems:

ODE-based (scipy.integrate):
- Reaction kinetics (consecutive reactions A -> B -> C)

Cantera-based (future, requires cantera):
- Ignition delay optimization
- Emission minimization (NOx, CO)
- Reactor design optimization

Examples
--------
>>> from surfaces.test_functions.simulation.chemical import ConsecutiveReactionFunction
>>> func = ConsecutiveReactionFunction(target_time=2.0)
>>> result = func({"k1": 1.0, "k2": 0.5})
"""

from .kinetics import ConsecutiveReactionFunction

__all__ = [
    "ConsecutiveReactionFunction",
]

chemical_functions = [
    ConsecutiveReactionFunction,
]
