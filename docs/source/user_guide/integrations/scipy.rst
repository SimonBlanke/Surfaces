.. _user_guide_scipy:

=====
scipy
=====

Surfaces provides built-in integration with ``scipy.optimize``.
Every test function has a ``to_scipy()`` method for seamless conversion.

----

Basic Usage
===========

.. code-block:: python

    from surfaces.test_functions.algebraic import RosenbrockFunction
    from scipy.optimize import minimize

    # Create test function
    func = RosenbrockFunction(n_dim=5)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run optimization
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    print(f"Minimum found at: {result.x}")
    print(f"Minimum value: {result.fun}")

----

The ``to_scipy()`` Method
=========================

Returns three values:

1. **objective**: Callable that takes a numpy array
2. **bounds**: List of (min, max) tuples for each dimension
3. **x0**: Initial point (center of search space)

.. code-block:: python

    objective, bounds, x0 = func.to_scipy()

    # objective is a function: np.ndarray -> float
    # bounds is: [(min0, max0), (min1, max1), ...]
    # x0 is: np.ndarray of starting point

----

Optimization Methods
====================

Local Optimizers
----------------

.. code-block:: python

    from scipy.optimize import minimize

    # Gradient-free methods
    result = minimize(objective, x0, bounds=bounds, method='Nelder-Mead')
    result = minimize(objective, x0, bounds=bounds, method='Powell')

    # Quasi-Newton methods
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

    # Trust region methods
    result = minimize(objective, x0, bounds=bounds, method='trust-constr')

Global Optimizers
-----------------

.. code-block:: python

    from scipy.optimize import differential_evolution, dual_annealing, basinhopping

    # Differential Evolution
    result = differential_evolution(objective, bounds, maxiter=1000)

    # Simulated Annealing
    result = dual_annealing(objective, bounds, maxiter=1000)

    # Basin Hopping
    result = basinhopping(objective, x0, minimizer_kwargs={'bounds': bounds})

----

Benchmarking Example
====================

Compare multiple scipy optimizers:

.. code-block:: python

    from surfaces.test_functions.algebraic import RastriginFunction
    from scipy.optimize import minimize, differential_evolution

    func = RastriginFunction(n_dim=10)
    objective, bounds, x0 = func.to_scipy()

    methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B']

    for method in methods:
        result = minimize(objective, x0, bounds=bounds, method=method)
        print(f"{method}: f={result.fun:.6f}, nfev={result.nfev}")

    # Global optimizer
    result = differential_evolution(objective, bounds, seed=42)
    print(f"DE: f={result.fun:.6f}, nfev={result.nfev}")

----

Limitations
===========

Continuous Parameters Only
--------------------------

``to_scipy()`` only works with continuous search spaces. ML functions
with categorical parameters cannot be converted:

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    func = KNeighborsClassifierFunction()

    # This will raise an error - categorical parameters
    # objective, bounds, x0 = func.to_scipy()  # Error!

For mixed parameter types, use Optuna or Hyperactive instead.

Local vs Global
---------------

Most scipy methods are local optimizers. For multimodal functions,
use global methods like ``differential_evolution`` or ``dual_annealing``.

----

Next Steps
==========

- :doc:`optuna` - For mixed parameter types and better HPO
- :doc:`/api_reference/base` - ``to_scipy()`` API reference
