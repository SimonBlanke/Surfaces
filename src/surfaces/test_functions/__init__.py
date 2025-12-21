# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

# Algebraic functions (always available - numpy only)
from .algebraic import *  # noqa: F401,F403
from .algebraic import algebraic_functions

# Machine learning functions (require sklearn)
from .machine_learning import machine_learning_functions

if machine_learning_functions:  # Only import if sklearn available
    from .machine_learning import *  # noqa: F401,F403

# Engineering functions (always available - numpy only)
from .engineering import *  # noqa: F401,F403
from .engineering import engineering_functions

# Backwards compatibility alias
mathematical_functions = algebraic_functions

test_functions: list = algebraic_functions + machine_learning_functions + engineering_functions
