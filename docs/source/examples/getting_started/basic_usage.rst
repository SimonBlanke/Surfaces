.. _example_basic_usage:

===========
Basic Usage
===========

Your first steps with Surfaces test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Creating a Test Function
========================

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Create a 3-dimensional Sphere function
    func = SphereFunction(n_dim=3)

    print(f"Function: {func}")
    print(f"Dimensions: {func.n_dim}")

----

Evaluating the Function
=======================

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=3)

    # Evaluate at a point using a dictionary
    params = {"x0": 1.0, "x1": 2.0, "x2": 3.0}
    result = func(params)

    print(f"f(1, 2, 3) = {result}")  # Output: 14.0

    # The global minimum is at the origin
    optimal = func({"x0": 0.0, "x1": 0.0, "x2": 0.0})
    print(f"f(0, 0, 0) = {optimal}")  # Output: 0.0

----

Loss for Minimization
=====================

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=3)

    # Default: returns loss (for minimization)
    loss = func({"x0": 1.0, "x1": 1.0, "x2": 1.0})
    print(f"Loss: {loss}")  # 3.0

    # For maximization, simply negate the result
    score = -func({"x0": 1.0, "x1": 1.0, "x2": 1.0})
    print(f"Score: {score}")  # -3.0

----

Complete Example
================

.. code-block:: python

    """Basic usage of Surfaces test functions."""

    from surfaces.test_functions.algebraic import SphereFunction, RastriginFunction

    def main():
        # Create functions
        sphere = SphereFunction(n_dim=3)
        rastrigin = RastriginFunction(n_dim=3)

        # Test point
        params = {"x0": 1.0, "x1": 1.0, "x2": 1.0}

        # Evaluate
        print(f"Sphere(1,1,1) = {sphere(params)}")
        print(f"Rastrigin(1,1,1) = {rastrigin(params)}")

        # Global optima (both at origin)
        origin = {"x0": 0.0, "x1": 0.0, "x2": 0.0}
        print(f"Sphere(0,0,0) = {sphere(origin)}")
        print(f"Rastrigin(0,0,0) = {rastrigin(origin)}")

    if __name__ == "__main__":
        main()

----

Next Steps
==========

- :doc:`input_formats` - Different ways to pass parameters
- :doc:`search_space` - Working with search spaces
