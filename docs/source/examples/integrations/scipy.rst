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

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize

    # Create test function
    func = RosenbrockFunction(n_dim=5)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Optimize with L-BFGS-B
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    print(f"Success: {result.success}")
    print(f"Minimum: {result.fun:.6f}")
    print(f"Evaluations: {result.nfev}")

----

Global Optimization
===================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from scipy.optimize import differential_evolution

    func = RastriginFunction(n_dim=10)
    objective, bounds, _ = func.to_scipy()

    # Differential Evolution for multimodal functions
    result = differential_evolution(
        objective,
        bounds,
        maxiter=1000,
        seed=42
    )

    print(f"Best value: {result.fun:.6f}")
    print(f"Evaluations: {result.nfev}")

----

Comparing scipy Methods
=======================

.. code-block:: python

    """Compare different scipy optimization methods."""

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize, differential_evolution

    func = RosenbrockFunction(n_dim=5)
    objective, bounds, x0 = func.to_scipy()

    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']

    print("Local methods:")
    for method in methods:
        result = minimize(objective, x0, bounds=bounds, method=method)
        print(f"  {method}: f={result.fun:.4f}, nfev={result.nfev}")

    print("\nGlobal method:")
    result = differential_evolution(objective, bounds, seed=42)
    print(f"  DE: f={result.fun:.6f}, nfev={result.nfev}")
