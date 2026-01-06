.. _example_input_formats:

=============
Input Formats
=============

Different ways to pass parameters to test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Dictionary Input
================

The primary and recommended interface:

.. code-block:: python

    from surfaces.test_functions.algebraic import AckleyFunction

    func = AckleyFunction()

    # Dictionary input
    result = func({"x0": 1.0, "x1": 2.0})
    print(f"Result: {result}")

----

Keyword Arguments
=================

.. code-block:: python

    from surfaces.test_functions.algebraic import AckleyFunction

    func = AckleyFunction()

    # Keyword arguments
    result = func(x0=1.0, x1=2.0)
    print(f"Result: {result}")

----

Comparison
==========

.. code-block:: python

    """Compare input formats."""

    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=3)

    # Both produce the same result
    r1 = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})
    r2 = func(x0=1.0, x1=2.0, x2=3.0)

    print(f"Dict:    {r1}")
    print(f"Kwargs:  {r2}")

    assert r1 == r2
    print("Both formats produce identical results!")
