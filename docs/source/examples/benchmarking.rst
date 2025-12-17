.. _examples_benchmarking:

============
Benchmarking
============

Examples of using Surfaces for algorithm benchmarking.

Comparing Algorithms on Multiple Functions
==========================================

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        RosenbrockFunction,
        AckleyFunction,
    )
    from scipy.optimize import minimize, differential_evolution
    import numpy as np
    import time

    # Define test functions
    functions = {
        'Sphere': SphereFunction(n_dim=5),
        'Rastrigin': RastriginFunction(n_dim=5),
        'Rosenbrock': RosenbrockFunction(n_dim=5),
        'Ackley': AckleyFunction(),
    }

    # Run benchmarks
    results = {}

    for name, func in functions.items():
        objective, bounds, x0 = func.to_scipy()
        lower, upper = func.get_bounds()
        bounds_list = list(zip(lower, upper))

        # L-BFGS-B (local)
        start = time.time()
        res_local = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        time_local = time.time() - start

        # Differential Evolution (global)
        start = time.time()
        res_global = differential_evolution(objective, bounds_list, seed=42, maxiter=100)
        time_global = time.time() - start

        results[name] = {
            'L-BFGS-B': {'value': res_local.fun, 'time': time_local},
            'DE': {'value': res_global.fun, 'time': time_global},
        }

    # Print results
    print("Function       | L-BFGS-B         | Diff. Evolution")
    print("-" * 55)
    for name, res in results.items():
        local = f"{res['L-BFGS-B']['value']:.4f} ({res['L-BFGS-B']['time']:.3f}s)"
        global_ = f"{res['DE']['value']:.4f} ({res['DE']['time']:.3f}s)"
        print(f"{name:14} | {local:16} | {global_}")

Statistical Benchmarking with Multiple Runs
===========================================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from scipy.optimize import differential_evolution
    import numpy as np

    func = RastriginFunction(n_dim=5)
    objective, _, _ = func.to_scipy()
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    # Run 30 independent trials
    n_trials = 30
    results = []

    for seed in range(n_trials):
        result = differential_evolution(
            objective,
            bounds_list,
            seed=seed,
            maxiter=200
        )
        results.append(result.fun)

    results = np.array(results)

    print(f"Trials: {n_trials}")
    print(f"Mean: {results.mean():.6f}")
    print(f"Std:  {results.std():.6f}")
    print(f"Min:  {results.min():.6f}")
    print(f"Max:  {results.max():.6f}")
    print(f"Success rate (< 0.1): {(results < 0.1).mean() * 100:.1f}%")

Scaling Analysis
================

.. code-block:: python

    from surfaces.test_functions import SphereFunction
    from scipy.optimize import minimize
    import time

    dimensions = [2, 5, 10, 20, 50, 100]

    print("Dim | Iterations | Time (s) | Final Value")
    print("-" * 45)

    for n_dim in dimensions:
        func = SphereFunction(n_dim=n_dim)
        objective, bounds, x0 = func.to_scipy()

        start = time.time()
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
        elapsed = time.time() - start

        print(f"{n_dim:3} | {result.nit:10} | {elapsed:8.4f} | {result.fun:.2e}")

Convergence Analysis
====================

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize
    import matplotlib.pyplot as plt

    func = RosenbrockFunction(n_dim=10)
    objective, bounds, x0 = func.to_scipy()

    # Track convergence
    history = []

    def callback(x):
        history.append(objective(x))

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        method='L-BFGS-B',
        callback=callback,
        options={'maxiter': 500}
    )

    # Plot convergence
    plt.figure(figsize=(10, 6))
    plt.semilogy(history)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Value (log scale)')
    plt.title('Convergence on Rosenbrock Function')
    plt.grid(True)
    plt.savefig('convergence.png')
    print("Saved convergence plot to convergence.png")

Function Evaluation Budget
==========================

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    from scipy.optimize import differential_evolution
    import numpy as np

    func = AckleyFunction()
    objective, _, _ = func.to_scipy()
    lower, upper = func.get_bounds()
    bounds_list = list(zip(lower, upper))

    # Different evaluation budgets
    budgets = [100, 500, 1000, 5000]
    n_trials = 10

    print("Budget | Mean Value | Success Rate")
    print("-" * 40)

    for budget in budgets:
        results = []
        for seed in range(n_trials):
            result = differential_evolution(
                objective,
                bounds_list,
                seed=seed,
                maxiter=budget // 20,  # Approximate iterations
                popsize=10
            )
            results.append(result.fun)

        results = np.array(results)
        success_rate = (results < 0.01).mean() * 100
        print(f"{budget:6} | {results.mean():10.4f} | {success_rate:5.1f}%")
