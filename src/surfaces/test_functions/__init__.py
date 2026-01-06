# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Test functions for optimization algorithm benchmarking.

Module Structure
----------------
algebraic/
    standard/       - Classic benchmark functions (Sphere, Rastrigin, etc.)
    constrained/    - Constrained problems (WeldedBeam, etc.)
    multi_objective/ - Multi-objective functions (ZDT, Kursawe, etc.)

benchmark/
    bbob/           - COCO/BBOB benchmark suite (24 functions)
    cec/            - CEC competition benchmarks (2013, 2014, 2017)

machine_learning/
    tabular/        - Tabular ML model HPO
    image/          - Image classification HPO
    timeseries/     - Time series HPO

simulation/
    structural/     - FEM/topology optimization (FEniCS)
    molecular/      - Molecular dynamics (OpenMM)
    chemical/       - Chemical kinetics (Cantera)
    electromagnetic/ - FDTD simulation (Meep)
    dynamics/       - Multibody dynamics (MuJoCo/PyBullet)

Note
----
Functions are NOT imported at this level to avoid dependency issues.
Import directly from submodules:

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.test_functions.benchmark.bbob import RosenbrockFunction
    from surfaces.test_functions.machine_learning.tabular import KNeighborsClassifierFunction
"""
