# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Test functions for optimization algorithm benchmarking.

Module Structure
----------------
algebraic/
    standard/       - Classic benchmark functions (Sphere, Rastrigin, etc.)
    constrained/    - Constrained problems (WeldedBeam, etc.)
    multi_objective/ - Multi-objective functions (ZDT, DTLZ, WFG, etc.)
    discrete/       - Binary / pseudo-boolean functions (OneMax, NK, Trap, etc.)

bbob/
    COCO/BBOB test function family (24 functions)

cec/
    CEC competition test function families (2013, 2014, 2017)

machine_learning/
    hyperparameter_optimization/ - Model HPO (tabular, image, time series)
    ensemble_optimization/       - Ensemble construction (stacking, voting)
    feature_engineering/         - Feature selection and transformation

simulation/
    chemical/        - Chemical reaction kinetics (scipy ODE)
    dynamics/        - Mechanical and population dynamics (scipy ODE)
    electromagnetic/ - Circuit simulation (scipy ODE)

Note
----
Functions are NOT imported at this level to avoid dependency issues.
The canonical import paths are the family namespaces; ML functions are
re-exported flat at ``machine_learning`` regardless of their category:

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.test_functions.bbob import RosenbrockOriginal
    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
"""
