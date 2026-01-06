# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Molecular dynamics simulation functions (OpenMM-based).

This module provides molecular simulation benchmarks including:
- Protein conformation optimization
- Molecular energy minimization
- Force field parameter optimization

Requirements
------------
- openmm
- numpy

Examples
--------
>>> from surfaces.test_functions.simulation.molecular import ConformationEnergy
>>> func = ConformationEnergy(molecule="alanine_dipeptide")
>>> result = func(dihedral_angles)
"""

__all__: list = []
molecular_functions: list = []
