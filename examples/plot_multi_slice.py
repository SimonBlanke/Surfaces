import os

from surfaces.test_functions.algebraic import SphereFunction

func = SphereFunction(n_dim=5)

# Using the accessor pattern
fig = func.plot.multi_slice()

# With custom range for one dimension
# fig = func.plot.multi_slice(params={"x0": (-2, 2), "x1": ..., "x2": ..., "x3": ..., "x4": ...})

# Fix one dimension, show slices for the rest
# fig = func.plot.multi_slice(params={"x2": 0.0})

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
