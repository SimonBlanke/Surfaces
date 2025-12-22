from surfaces.test_functions.algebraic.test_functions_nd import RastriginFunction
from surfaces.visualize import plot_convergence

func = RastriginFunction(n_dim=10)
history = [120, 95, 80, 65, 50, 42, 35, 28, 22, 18, 15, 12, 10, 8, 6, 5, 4, 3]
fig = plot_convergence(func, history)
fig.show()
