import os

from surfaces.test_functions.algebraic import AckleyFunction

# 2D function - simple case
func = AckleyFunction()
fig = func.plot.surface()

# N-D function - select which dimensions to plot
# func = SphereFunction(n_dim=5)
# fig = func.plot.surface(params={"x0": ..., "x2": ...})

# Custom ranges for the axes
# fig = func.plot.surface(params={"x0": (-2, 2), "x1": (-1, 1)})

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
