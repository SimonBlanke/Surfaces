from surfaces.test_functions.algebraic.test_functions_nd import SphereFunction
from surfaces.visualize import plot_multi_slice

func = SphereFunction(n_dim=5)
fig = plot_multi_slice(func)
fig.show()
