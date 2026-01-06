# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Simulation-based test functions.

This module provides optimization benchmarks based on numerical simulations.
Unlike algebraic functions, these involve actual physics simulations.

Available Now (scipy.integrate, no extra deps)
----------------------------------------------
dynamics:
    - LotkaVolterraFunction: Predator-prey population dynamics
    - DampedOscillatorFunction: Mechanical oscillator optimization
chemical:
    - ConsecutiveReactionFunction: A -> B -> C reaction kinetics
electromagnetic:
    - RLCCircuitFunction: Series RLC circuit optimization
    - RCFilterFunction: Low-pass filter design

Future (requires external packages)
-----------------------------------
structural : FEniCS-based structural mechanics (FEM, topology optimization)
molecular : OpenMM-based molecular dynamics
chemical : Cantera-based combustion (in addition to scipy-based kinetics)
electromagnetic : Meep-based FDTD electromagnetic simulation
dynamics : MuJoCo/PyBullet-based multibody dynamics and robotics

Note
----
Functions will raise ImportError with installation instructions if
dependencies are missing.

Examples
--------
>>> from surfaces.test_functions.simulation import LotkaVolterraFunction
>>> func = LotkaVolterraFunction()
>>> result = func({"alpha": 1.0, "beta": 0.1, "gamma": 1.5, "delta": 0.075})
"""

from ._base_simulation import SimulationFunction

# ODE-based functions (scipy only, always available)
from .chemical import ConsecutiveReactionFunction, chemical_functions
from .dynamics import (
    DampedOscillatorFunction,
    LotkaVolterraFunction,
    ODESimulationFunction,
    dynamics_functions,
)
from .electromagnetic import (
    RCFilterFunction,
    RLCCircuitFunction,
    electromagnetic_functions,
)

__all__ = [
    # Base classes
    "SimulationFunction",
    "ODESimulationFunction",
    # Dynamics
    "LotkaVolterraFunction",
    "DampedOscillatorFunction",
    # Chemical
    "ConsecutiveReactionFunction",
    # Electromagnetic
    "RLCCircuitFunction",
    "RCFilterFunction",
    # Function lists
    "simulation_functions",
]

# Combined list of all simulation functions
simulation_functions: list = dynamics_functions + chemical_functions + electromagnetic_functions
