# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Simulation-based test functions.

This module provides optimization benchmarks based on numerical simulations.
Unlike algebraic functions, these involve actual physics simulations and have
significant evaluation time (seconds to minutes).

Submodules
----------
structural : FEniCS-based structural mechanics (FEM, topology optimization)
molecular : OpenMM-based molecular dynamics
chemical : Cantera-based chemical kinetics and combustion
electromagnetic : Meep-based FDTD electromagnetic simulation
dynamics : MuJoCo/PyBullet-based multibody dynamics and robotics

Note
----
Each submodule requires specific external dependencies. Functions will raise
ImportError with installation instructions if dependencies are missing.
"""

from ._base_simulation import SimulationFunction

__all__ = [
    "SimulationFunction",
]

# Simulation functions list (populated by submodules when available)
simulation_functions: list = []
