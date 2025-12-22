from surfaces.test_functions.algebraic.test_functions_2d import BealeFunction
from surfaces.visualize import plot_contour

func = BealeFunction()
fig = plot_contour(func)
fig.show()
