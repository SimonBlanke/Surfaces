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

    from surfaces.test_functions.algebraic import WeldedBeamFunction

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

    from surfaces.test_functions.algebraic import WeldedBeamFunction

    func = WeldedBeamFunction()
    params = {"h": 0.2, "l": 3.5, "t": 9.0, "b": 0.2}

    # Get constraint values (g <= 0 is feasible)
    constraints = func.constraints(params)
    print("Constraints (g <= 0 is OK):")
    for i, g in enumerate(constraints):
        status = "OK" if g <= 0 else "VIOLATED"
        print(f"  g{i+1}: {g:.4f} [{status}]")

----

Exploring the Design Space
==========================

.. code-block:: python

    import random
    from surfaces.test_functions.algebraic import WeldedBeamFunction

    func = WeldedBeamFunction()
    space = func.search_space

    print("Design space bounds:")
    for name, values in space.items():
        print(f"  {name}: [{min(values):.3f}, {max(values):.3f}]")

    # Random search for feasible designs
    print("\nSearching for feasible designs...")
    feasible_count = 0
    for _ in range(100):
        sample = {k: random.choice(v) for k, v in space.items()}
        if func.is_feasible(sample):
            feasible_count += 1
            cost = func.raw_objective(sample)
            print(f"  Feasible design found, cost: {cost:.4f}")
            break

    print(f"Feasible designs in 100 samples: {feasible_count}")
