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

    # Get the default search space
    space = func.search_space()

    print(f"Parameters: {list(space.keys())}")
    # Output: ['x0', 'x1', 'x2']

    for name, values in space.items():
        print(f"{name}: [{values.min():.2f}, {values.max():.2f}]")

----

Sampling from Search Space
==========================

.. code-block:: python

    # Get a random sample from the search space
    sample = func.search_space_sample()
    print(f"Random sample: {sample}")

    # Evaluate at the sample
    result = func(sample)
    print(f"Result: {result}")

----

Custom Search Space Resolution
==============================

.. code-block:: python

    # Default resolution
    space_default = func.search_space()
    print(f"Default points per dimension: {len(space_default['x0'])}")

    # Custom resolution
    space_fine = func.search_space(resolution=1000)
    print(f"Fine points per dimension: {len(space_fine['x0'])}")

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
        space = func.search_space()

        print("Search Space:")
        for name, values in space.items():
            print(f"  {name}: [{values.min():.1f}, {values.max():.1f}]")

        # Random sampling
        print("\nRandom samples:")
        for i in range(5):
            sample = func.search_space_sample()
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
