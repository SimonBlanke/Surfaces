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

    from surfaces.test_functions import AckleyFunction

    func = AckleyFunction()

    # Dictionary input
    result = func({"x0": 1.0, "x1": 2.0})
    print(f"Result: {result}")

----

Keyword Arguments
=================

.. code-block:: python

    # Keyword arguments
    result = func(x0=1.0, x1=2.0)
    print(f"Result: {result}")

----

Positional Arguments
====================

Using the ``evaluate()`` method:

.. code-block:: python

    # Positional arguments via evaluate()
    result = func.evaluate(1.0, 2.0)
    print(f"Result: {result}")

----

NumPy Array
===========

Using the ``evaluate_array()`` method:

.. code-block:: python

    import numpy as np

    # NumPy array input
    x = np.array([1.0, 2.0])
    result = func.evaluate_array(x)
    print(f"Result: {result}")

----

Comparison
==========

.. code-block:: python

    """Compare all input formats."""

    import numpy as np
    from surfaces.test_functions import SphereFunction

    func = SphereFunction(n_dim=3)

    # All produce the same result
    r1 = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})
    r2 = func(x0=1.0, x1=2.0, x2=3.0)
    r3 = func.evaluate(1.0, 2.0, 3.0)
    r4 = func.evaluate_array(np.array([1.0, 2.0, 3.0]))

    print(f"Dict:    {r1}")
    print(f"Kwargs:  {r2}")
    print(f"Positional: {r3}")
    print(f"Array:   {r4}")

    assert r1 == r2 == r3 == r4
    print("All formats produce identical results!")
