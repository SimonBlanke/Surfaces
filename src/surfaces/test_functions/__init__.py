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
"""

# Algebraic functions (always available - numpy only)
# Includes: standard, constrained (formerly engineering), multi_objective
from .algebraic import *  # noqa: F401,F403
from .algebraic import algebraic_functions, constrained_functions, standard_functions

# Benchmark functions (BBOB always available, CEC requires data package)
from .benchmark import *  # noqa: F401,F403
from .benchmark import bbob_functions, benchmark_functions

# Machine learning functions (require sklearn)
from .machine_learning import machine_learning_functions

if machine_learning_functions:  # Only import if sklearn available
    from .machine_learning import *  # noqa: F401,F403

# Simulation functions (base class always available, implementations require deps)
from .simulation import SimulationFunction, simulation_functions

# Backwards compatibility aliases
mathematical_functions = algebraic_functions
engineering_functions = constrained_functions

# Combined list of all available test functions
test_functions: list = (
    algebraic_functions + benchmark_functions + machine_learning_functions + simulation_functions
)
