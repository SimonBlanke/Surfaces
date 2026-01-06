import os

from surfaces.test_functions.algebraic import AckleyFunction
from surfaces.visualize import plot_surface

func = AckleyFunction()
fig = plot_surface(func)

if not os.environ.get("SURFACES_TESTING"):
    fig.show()
