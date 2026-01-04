.. _example_engineering:

=====================
Engineering Functions
=====================

Examples using constrained engineering design problems.

.. contents:: On this page
   :local:
   :depth: 2

----

Welded Beam Design
==================

.. code-block:: python

    from surfaces.test_functions import WeldedBeamFunction

    func = WeldedBeamFunction()

    # Design parameters
    params = {
        "h": 0.2,    # Weld thickness
        "l": 3.5,    # Weld length
        "t": 9.0,    # Beam thickness
        "b": 0.2     # Beam width
    }

    # Penalized objective (includes constraint violations)
    cost = func(params)
    print(f"Penalized cost: {cost:.4f}")

    # Raw objective without penalty
    raw_cost = func.raw_objective(params)
    print(f"Raw cost: {raw_cost:.4f}")

    # Check feasibility
    print(f"Feasible: {func.is_feasible(params)}")

----

Constraint Handling
===================

.. code-block:: python

    func = WeldedBeamFunction()
    params = {"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2}

    # Get constraint values (g <= 0 is feasible)
    constraints = func.constraints(params)
    print("Constraints (g <= 0 is OK):")
    for i, g in enumerate(constraints):
        status = "OK" if g <= 0 else "VIOLATED"
        print(f"  g{i+1}: {g:.4f} [{status}]")

    # Get violation amounts
    violations = func.constraint_violations(params)
    print(f"\nTotal violation: {sum(violations):.4f}")

----

Optimizing Engineering Problems
===============================

.. code-block:: python

    from surfaces.test_functions import WeldedBeamFunction
    from scipy.optimize import differential_evolution

    func = WeldedBeamFunction()

    # Get scipy format
    objective, bounds, x0 = func.to_scipy()

    # Optimize
    result = differential_evolution(
        objective,
        bounds,
        seed=42,
        maxiter=500
    )

    # Convert back to parameters
    params = {
        "h": result.x[0],
        "l": result.x[1],
        "t": result.x[2],
        "b": result.x[3]
    }

    print(f"Optimal cost: {func.raw_objective(params):.4f}")
    print(f"Feasible: {func.is_feasible(params)}")
    print(f"Parameters: {params}")
