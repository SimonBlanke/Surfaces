.. _examples_scipy_optimization:

==================
scipy Optimization
==================

Examples of using Surfaces with scipy.optimize.

Basic Optimization with L-BFGS-B
================================

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize

    # Create a 3D Rosenbrock function
    func = RosenbrockFunction(n_dim=3)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run optimization
    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B',
        options={'maxiter': 1000}
    )

    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print(f"Optimal x: {result.x}")
    print(f"Minimum value: {result.fun}")

Comparing Local Optimizers
==========================

.. code-block:: python

    from surfaces.test_functions import SphereFunction
    from scipy.optimize import minimize
    import numpy as np

    func = SphereFunction(n_dim=5)
    objective, bounds, x0 = func.to_scipy()

    methods = ['L-BFGS-B', 'SLSQP', 'trust-constr']

    for method in methods:
        result = minimize(objective, x0, bounds=bounds, method=method)
        print(f"{method}: f(x) = {result.fun:.6f}, iters = {result.nit}")

Global Optimization with Differential Evolution
===============================================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from scipy.optimize import differential_evolution

    # Rastrigin has many local minima
    func = RastriginFunction(n_dim=5)
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    # Get scipy-compatible objective
    objective, _, _ = func.to_scipy()

    # Run differential evolution
    result = differential_evolution(
        objective,
        bounds_list,
        maxiter=1000,
        seed=42
    )

    print(f"Found minimum: {result.fun:.6f}")
    print(f"At position: {result.x}")
    print(f"Global minimum is 0 at origin")

Dual Annealing for Multimodal Functions
=======================================

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    from scipy.optimize import dual_annealing

    func = AckleyFunction()
    objective, _, _ = func.to_scipy()
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    result = dual_annealing(
        objective,
        bounds_list,
        maxiter=1000,
        seed=42
    )

    print(f"Found minimum: {result.fun:.6f}")
    print(f"True minimum: 0 at (0, 0)")

Basin Hopping with Local Refinement
===================================

.. code-block:: python

    from surfaces.test_functions import HimmelblausFunction
    from scipy.optimize import basinhopping
    import numpy as np

    func = HimmelblausFunction()
    objective, bounds, x0 = func.to_scipy()

    # Custom minimizer kwargs
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds
    }

    # Run basin hopping
    np.random.seed(42)
    result = basinhopping(
        objective,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=100
    )

    print(f"Found minimum: {result.fun:.6f}")
    print(f"At position: {result.x}")

Optimization with Constraints
=============================

.. code-block:: python

    from surfaces.test_functions import SphereFunction
    from scipy.optimize import minimize, NonlinearConstraint
    import numpy as np

    func = SphereFunction(n_dim=2)
    objective, bounds, x0 = func.to_scipy()

    # Add a constraint: x0 + x1 >= 1
    def constraint_func(x):
        return x[0] + x[1]

    constraint = NonlinearConstraint(constraint_func, 1.0, np.inf)

    result = minimize(
        objective,
        x0=[0.5, 0.5],
        bounds=bounds,
        constraints=constraint,
        method='SLSQP'
    )

    print(f"Optimal x: {result.x}")
    print(f"Constraint value: {result.x[0] + result.x[1]:.4f}")  # Should be >= 1

Tracking Optimization Progress
==============================

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize

    func = RosenbrockFunction(n_dim=3)
    objective, bounds, x0 = func.to_scipy()

    # Track history
    history = []

    def callback(x):
        history.append(objective(x))

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B',
        callback=callback
    )

    print(f"Optimization took {len(history)} iterations")
    print(f"Initial value: {history[0]:.4f}")
    print(f"Final value: {history[-1]:.6f}")
