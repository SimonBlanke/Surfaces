"""Backward compatibility alias.

The canonical location is now
``surfaces.test_functions._base_multi_objective.BaseMultiObjectiveTestFunction``.
"""

from ..._base_multi_objective import BaseMultiObjectiveTestFunction

# Backward compatibility
MultiObjectiveFunction = BaseMultiObjectiveTestFunction
