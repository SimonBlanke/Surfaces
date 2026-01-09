import os

from surfaces.test_functions.algebraic import BealeFunction

# Using the accessor pattern
func = BealeFunction()
fig = func.plot.contour()

# Custom ranges for the axes
# fig = func.plot.contour(params={"x0": (-2, 2), "x1": (-1, 1)})

# Heatmap is also available
# fig = func.plot.heatmap()

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
