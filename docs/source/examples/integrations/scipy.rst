.. _example_scipy:

=====
scipy
=====

Examples using Surfaces with scipy.optimize.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic scipy Integration
=======================

.. code-block:: python

    import numpy as np
    from surfaces.test_functions.algebraic import RosenbrockFunction
    from scipy.optimize import minimize

    # Create test function
    func = RosenbrockFunction(n_dim=5)
    space = func.search_space

    # Convert to scipy format
    def objective(x):
        params = {f"x{i}": x[i] for i in range(len(x))}
        return func(params)

    bounds = [(v.min(), v.max()) for v in space.values()]
    x0 = np.array([np.mean(v) for v in space.values()])

    # Optimize with L-BFGS-B
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    print(f"Success: {result.success}")
    print(f"Minimum: {result.fun:.6f}")
    print(f"Evaluations: {result.nfev}")

----

Global Optimization
===================

.. code-block:: python

    import numpy as np
    from surfaces.test_functions.algebraic import RastriginFunction
    from scipy.optimize import differential_evolution

    func = RastriginFunction(n_dim=10)
    space = func.search_space

    def objective(x):
        params = {f"x{i}": x[i] for i in range(len(x))}
        return func(params)

    bounds = [(v.min(), v.max()) for v in space.values()]

    # Differential Evolution for multimodal functions
    result = differential_evolution(
        objective,
        bounds,
        maxiter=500,
        seed=42
    )

    print(f"Best value: {result.fun:.6f}")
    print(f"Evaluations: {result.nfev}")

----

Comparing scipy Methods
=======================

.. code-block:: python

    """Compare different scipy optimization methods."""

    import numpy as np
    from surfaces.test_functions.algebraic import RosenbrockFunction
    from scipy.optimize import minimize, differential_evolution

    func = RosenbrockFunction(n_dim=5)
    space = func.search_space

    def objective(x):
        params = {f"x{i}": x[i] for i in range(len(x))}
        return func(params)

    bounds = [(v.min(), v.max()) for v in space.values()]
    x0 = np.array([np.mean(v) for v in space.values()])

    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']

    print("Local methods:")
    for method in methods:
        result = minimize(objective, x0, bounds=bounds, method=method)
        print(f"  {method}: f={result.fun:.4f}, nfev={result.nfev}")

    print("\nGlobal method:")
    result = differential_evolution(objective, bounds, seed=42, maxiter=200)
    print(f"  DE: f={result.fun:.6f}, nfev={result.nfev}")
