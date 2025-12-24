# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Noise layers for adding stochastic disturbances to test functions.

This module provides noise classes that can be passed to test functions
to simulate noisy evaluations. Useful for testing algorithm robustness
to measurement uncertainty.

Examples
--------
Basic usage with a test function:

>>> from surfaces.test_functions import SphereFunction
>>> from surfaces.noise import GaussianNoise
>>>
>>> noise = GaussianNoise(sigma=0.1, seed=42)
>>> func = SphereFunction(n_dim=2, noise=noise)
>>> result = func([0.5, 0.5])  # Returns noisy evaluation

Decaying noise over optimization:

>>> noise = GaussianNoise(
...     sigma=0.5,
...     sigma_final=0.01,
...     schedule="linear",
...     total_evaluations=1000,
...     seed=42
... )
>>> func = SphereFunction(n_dim=2, noise=noise)

Available noise types:

- GaussianNoise: Additive Gaussian noise, f(x) + N(0, sigma^2)
- UniformNoise: Additive uniform noise, f(x) + U(low, high)
- MultiplicativeNoise: Multiplicative noise, f(x) * (1 + N(0, sigma^2))
"""

from ._base import BaseNoise
from ._gaussian import GaussianNoise
from ._multiplicative import MultiplicativeNoise
from ._uniform import UniformNoise

__all__ = [
    "BaseNoise",
    "GaussianNoise",
    "UniformNoise",
    "MultiplicativeNoise",
]
