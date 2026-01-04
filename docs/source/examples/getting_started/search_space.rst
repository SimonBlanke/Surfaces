.. _example_search_space:

============
Search Space
============

Working with search spaces and parameter bounds.

.. contents:: On this page
   :local:
   :depth: 2

----

Getting the Search Space
========================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=3)

    # Get the default search space (property, not method)
    space = func.search_space

    print(f"Parameters: {list(space.keys())}")
    # Output: ['x0', 'x1', 'x2']

    for name, values in space.items():
        print(f"{name}: [{values.min():.2f}, {values.max():.2f}]")

----

Sampling from Search Space
==========================

.. code-block:: python

    import numpy as np
    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=3)
    space = func.search_space

    # Get a random sample from the search space
    sample = {k: np.random.choice(v) for k, v in space.items()}
    print(f"Random sample: {sample}")

    # Evaluate at the sample
    result = func(sample)
    print(f"Result: {result}")

----

Search Space Structure
======================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction

    func = RastriginFunction(n_dim=3)
    space = func.search_space

    # Each dimension contains an array of allowed values
    print(f"Points per dimension: {len(space['x0'])}")
    print(f"x0 range: [{space['x0'].min():.2f}, {space['x0'].max():.2f}]")

----

Complete Example
================

.. code-block:: python

    """Working with search spaces."""

    import numpy as np
    from surfaces.test_functions import RosenbrockFunction

    def main():
        func = RosenbrockFunction(n_dim=5)

        # Get search space
        space = func.search_space

        print("Search Space:")
        for name, values in space.items():
            print(f"  {name}: [{values.min():.1f}, {values.max():.1f}]")

        # Random sampling
        print("\nRandom samples:")
        for i in range(5):
            sample = {k: np.random.choice(v) for k, v in space.items()}
            result = func(sample)
            print(f"  Sample {i+1}: f = {result:.4f}")

        # Grid evaluation
        print("\nGrid evaluation (first 2 dims):")
        x0_values = np.linspace(-2, 2, 5)
        x1_values = np.linspace(-2, 2, 5)

        for x0 in x0_values:
            for x1 in x1_values:
                params = {f"x{i}": 0.0 for i in range(5)}
                params["x0"] = x0
                params["x1"] = x1
                result = func(params)
                print(f"  f({x0:.1f}, {x1:.1f}, 0, 0, 0) = {result:.2f}")

    if __name__ == "__main__":
        main()
