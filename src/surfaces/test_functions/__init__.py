# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

# Algebraic functions (always available - numpy only)
# Includes: standard, constrained (formerly engineering), multi_objective
from .algebraic import *  # noqa: F401,F403
from .algebraic import algebraic_functions, constrained_functions

# Machine learning functions (require sklearn)
from .machine_learning import machine_learning_functions

if machine_learning_functions:  # Only import if sklearn available
    from .machine_learning import *  # noqa: F401,F403

# Backwards compatibility aliases
mathematical_functions = algebraic_functions
engineering_functions = constrained_functions  # Backwards compat

test_functions: list = algebraic_functions + machine_learning_functions
