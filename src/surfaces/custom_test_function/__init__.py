# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Custom test function module for user-defined objectives.

This module provides the CustomTestFunction class that wraps user-defined
objective functions with rich analysis, visualization, and persistence features.

Example
-------
>>> from surfaces.custom_test_function import CustomTestFunction
>>>
>>> def my_objective(params):
...     return params["x"]**2 + params["y"]**2
>>>
>>> func = CustomTestFunction(
...     objective_fn=my_objective,
...     search_space={"x": (-5, 5), "y": (-5, 5)},
...     experiment="my-experiment",
... )
>>>
>>> # Use with any optimizer
>>> score = func({"x": 1.0, "y": 2.0})
>>>
>>> # Analyze results
>>> func.analysis.summary()
>>> func.analysis.parameter_importance()
>>>
>>> # Visualize
>>> func.plot.history()
>>> func.plot.contour("x", "y")
>>>
>>> # Build surrogate
>>> func.surrogate.fit()
>>> func.surrogate.predict({"x": 0.5, "y": 0.5})

With persistence:

>>> from surfaces.custom_test_function import CustomTestFunction, SQLiteStorage
>>>
>>> storage = SQLiteStorage("./experiments.db", experiment="my-exp")
>>> func = CustomTestFunction(
...     objective_fn=my_objective,
...     search_space={"x": (-5, 5), "y": (-5, 5)},
...     storage=storage,
... )
>>>
>>> # Evaluations are automatically persisted
>>> func({"x": 1.0, "y": 2.0})
>>>
>>> # Storage operations via namespace
>>> func.storage.save_checkpoint()
>>> best = func.storage.query(order_by="score", limit=10)
>>> func.storage.delete()  # Clear all data
"""

from ._custom_test_function import CustomTestFunction
from .storage import InMemoryStorage, SQLiteStorage, Storage

__all__ = [
    "CustomTestFunction",
    "Storage",
    "InMemoryStorage",
    "SQLiteStorage",
]
