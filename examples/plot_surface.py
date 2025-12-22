from surfaces.test_functions.algebraic.test_functions_2d import AckleyFunction
from surfaces.visualize import plot_surface

func = AckleyFunction()
fig = plot_surface(func)
fig.show()
