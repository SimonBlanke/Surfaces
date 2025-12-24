# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Pre-defined function presets for standardized optimizer testing.

This module provides curated collections of test functions for different
benchmarking scenarios. Using standardized presets enables:

- Comparable results across different papers and projects
- Appropriate function selection for specific use cases
- Reduced boilerplate when setting up benchmarks

Available Presets
-----------------
quick : list
    5 functions for fast sanity checks during development.

standard : list
    15 well-known functions covering diverse landscape types.

algebraic_2d : list
    All 2D algebraic functions (18 functions).

algebraic_nd : list
    N-dimensional scalable functions (5 functions).

bbob : list
    Full BBOB/COCO benchmark (24 functions).

cec2014 : list
    CEC 2014 competition functions (30 functions).

cec2017 : list
    CEC 2017 simple functions (10 functions).

engineering : list
    Constrained engineering design problems (5 functions).

Examples
--------
2D functions instantiate directly:

>>> from surfaces.presets import algebraic_2d
>>> for FuncClass in algebraic_2d:
...     func = FuncClass()
...     result = optimizer.minimize(func)

N-dimensional functions require n_dim parameter:

>>> from surfaces.presets import algebraic_nd
>>> for FuncClass in algebraic_nd:
...     func = FuncClass(n_dim=10)
...     result = optimizer.minimize(func)

Use instantiate() for mixed presets:

>>> from surfaces.presets import standard, instantiate
>>> functions = instantiate(standard, n_dim=10)
>>> for func in functions:
...     result = optimizer.minimize(func)

Notes
-----
All presets contain function **classes**, not instances. This allows
customization of parameters (n_dim, instance, etc.) when instantiating.

Function classes fall into two categories:

1. **Fixed-dimension** (algebraic_2d, engineering): Instantiate with FuncClass().
2. **Scalable** (algebraic_nd, bbob, cec2014, cec2017): Require FuncClass(n_dim=N).

The `quick` and `standard` presets contain a mix of both types.
Use `instantiate()` to handle this automatically.
"""

from .suites import (
    algebraic_2d,
    algebraic_nd,
    bbob,
    cec2014,
    cec2017,
    engineering,
    quick,
    standard,
)
from .utilities import (
    get,
    instantiate,
    list_presets,
)

__all__ = [
    # Presets
    "quick",
    "standard",
    "algebraic_2d",
    "algebraic_nd",
    "bbob",
    "cec2014",
    "cec2017",
    "engineering",
    # Utilities
    "instantiate",
    "get",
    "list_presets",
]
