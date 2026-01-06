# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Function collection for discovering and filtering test functions.

This module provides a unified interface to browse, filter, and select
test functions from the Surfaces library.

Examples
--------
>>> from surfaces import collection
>>>
>>> # collection is iterable and contains all functions
>>> len(collection)                       # Total number of functions
>>> for func_cls in collection:
...     print(func_cls.__name__)
>>>
>>> # Filter and search
>>> collection.filter(unimodal=True)      # Filter by properties
>>> collection.filter(category="algebraic")
>>> collection.search("rastrigin")        # Search in names/taglines
>>>
>>> # Predefined suites
>>> collection.quick                      # 5 functions for smoke tests
>>> collection.standard                   # 15 functions for academic comparison
>>> collection.bbob                       # 24 COCO/BBOB functions
>>> collection.engineering                # 5 constrained problems
>>>
>>> # Iterate and instantiate
>>> for func_cls in collection.quick:
...     func = func_cls(n_dim=10)         # ND functions need n_dim
...     result = func({"x0": 0.0, ...})
"""

from .singleton import _CollectionSingleton

# Module-level singleton instance
collection = _CollectionSingleton()

__all__ = ["collection"]
