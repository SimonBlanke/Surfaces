# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Electromagnetic simulation functions (Meep-based).

This module provides FDTD electromagnetic benchmarks including:
- Waveguide optimization
- Antenna design
- Photonic device optimization

Requirements
------------
- meep
- numpy

Examples
--------
>>> from surfaces.test_functions.simulation.electromagnetic import WaveguideSplitter
>>> func = WaveguideSplitter(wavelength=1.55, resolution=20)
>>> result = func(design_params)
"""

__all__: list = []
electromagnetic_functions: list = []
