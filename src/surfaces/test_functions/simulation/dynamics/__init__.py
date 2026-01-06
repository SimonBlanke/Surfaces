# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Dynamical systems simulation functions.

This module provides optimization benchmarks based on dynamical systems:

ODE-based (scipy.integrate):
- Population dynamics (Lotka-Volterra predator-prey)
- Mechanical systems (damped oscillator)

Physics engines (future, requires mujoco/pybullet):
- Gait optimization for legged robots
- Control parameter tuning
- Trajectory optimization

Examples
--------
>>> from surfaces.test_functions.simulation.dynamics import LotkaVolterraFunction
>>> func = LotkaVolterraFunction()
>>> result = func({"alpha": 1.0, "beta": 0.1, "gamma": 1.5, "delta": 0.075})
"""

from ._base_ode import ODESimulationFunction
from .mechanical import DampedOscillatorFunction
from .population import LotkaVolterraFunction

__all__ = [
    "ODESimulationFunction",
    "LotkaVolterraFunction",
    "DampedOscillatorFunction",
]

dynamics_functions = [
    LotkaVolterraFunction,
    DampedOscillatorFunction,
]
