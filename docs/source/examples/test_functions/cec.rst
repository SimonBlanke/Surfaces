.. _example_cec:

=============
CEC Functions
=============

Examples using CEC competition benchmark functions.

.. contents:: On this page
   :local:
   :depth: 2

----

CEC 2013 Functions
==================

.. code-block:: python

    import numpy as np
    from surfaces.test_functions.benchmark.cec.cec2013 import (
        RotatedRastrigin,
        RotatedBentCigar,
        CompositionFunction1,
    )

    # Create functions
    functions = [
        RotatedRastrigin(n_dim=10),
        RotatedBentCigar(n_dim=10),
        CompositionFunction1(n_dim=10),
    ]

    for func in functions:
        space = func.search_space
        sample = {k: np.random.choice(v) for k, v in space.items()}
        result = func(sample)
        print(f"{func.__class__.__name__}: {result:.4f}")

----

Shifted Optima
==============

.. code-block:: python

    """Demonstrate that CEC optima are NOT at the origin."""

    from surfaces.test_functions.benchmark.cec.cec2013 import RotatedRastrigin
    from surfaces.test_functions.algebraic import RastriginFunction

    # Standard Rastrigin - optimum at origin
    standard = RastriginFunction(n_dim=5)
    origin = {f"x{i}": 0.0 for i in range(5)}
    print(f"Standard Rastrigin at origin: {standard(origin):.6f}")

    # CEC Rastrigin - optimum is shifted
    cec = RotatedRastrigin(n_dim=5)
    print(f"CEC Rastrigin at origin: {cec(origin):.6f}")
    print("(Not the minimum - optimum is shifted!)")

----

Composition Functions
=====================

.. code-block:: python

    """CEC composition functions combine multiple landscapes."""

    import numpy as np
    from surfaces.test_functions.benchmark.cec.cec2013 import (
        CompositionFunction1,
        CompositionFunction2,
        CompositionFunction3,
    )

    compositions = [
        CompositionFunction1(n_dim=10),
        CompositionFunction2(n_dim=10),
        CompositionFunction3(n_dim=10),
    ]

    for func in compositions:
        space = func.search_space
        results = []
        for _ in range(100):
            sample = {k: np.random.choice(v) for k, v in space.items()}
            results.append(func(sample))
        print(f"{func.__class__.__name__}:")
        print(f"  Min: {min(results):.2f}, Max: {max(results):.2f}")
