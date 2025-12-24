.. _user_guide_engineering:

=====================
Engineering Functions
=====================

Surfaces provides a collection of classic engineering design optimization
problems. These are real-world inspired benchmarks with physical meaning,
commonly used to evaluate constrained optimization algorithms.

Overview
========

Engineering functions differ from algebraic test functions in several ways:

- They have **physical meaning** (minimizing weight, cost, etc.)
- They include **constraints** that must be satisfied
- The search space represents **design parameters** with real-world bounds

All engineering functions use penalty methods to handle constraints. The
objective function returns the penalized value by default.

Available Functions
===================

The following engineering design functions are available:

.. include:: /_generated/catalogs/engineering.rst
   :start-after: Engineering Design
   :end-before: .. list-table::

.. include:: /_generated/catalogs/engineering.rst
   :start-after: Real-world constrained

Constraint Handling
===================

Each engineering function provides methods to work with constraints:

.. code-block:: python

    from surfaces.test_functions import WeldedBeamFunction

    func = WeldedBeamFunction()

    params = {"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2}

    # Get the penalized objective (default)
    penalized_cost = func(params)

    # Get the raw objective without penalty
    raw_cost = func.raw_objective(params)

    # Check constraint values (g <= 0 is feasible)
    constraints = func.constraints(params)

    # Check if solution is feasible
    is_ok = func.is_feasible(params)

    # Get violation amounts (positive = violated)
    violations = func.constraint_violations(params)

Penalty Coefficient
-------------------

The penalty coefficient can be adjusted when creating the function:

.. code-block:: python

    # Higher penalty = stronger constraint enforcement
    func = WeldedBeamFunction(penalty_coefficient=1e6)

    # Lower penalty = softer constraint handling
    func = WeldedBeamFunction(penalty_coefficient=1e3)

Example: Welded Beam Design
===========================

The welded beam problem minimizes fabrication cost while satisfying
stress, deflection, and buckling constraints:

.. code-block:: python

    from surfaces.test_functions import WeldedBeamFunction
    from scipy.optimize import differential_evolution

    # Create the function
    func = WeldedBeamFunction()

    # Get scipy-compatible interface
    objective, bounds, x0 = func.to_scipy()

    # Optimize
    result = differential_evolution(objective, bounds, seed=42)

    # Check feasibility of solution
    params = {"h": result.x[0], "l": result.x[1],
              "t": result.x[2], "b": result.x[3]}

    print(f"Cost: {func.raw_objective(params):.4f}")
    print(f"Feasible: {func.is_feasible(params)}")

Function Reference
==================

For complete API documentation of each function, see the
:doc:`API Reference </api_reference>`.
