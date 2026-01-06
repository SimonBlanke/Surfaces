"""Example: Using the Sphere function at different dimensions."""

from surfaces.test_functions.algebraic import SphereFunction

# 1D Sphere function
sphere_1d = SphereFunction(n_dim=1)
search_space = sphere_1d.search_space
result = sphere_1d({"x0": 0.5})
print(f"1D Sphere at x0=0.5: {result}")

# 2D Sphere function
sphere_2d = SphereFunction(n_dim=2)
search_space = sphere_2d.search_space
result = sphere_2d({"x0": 1.0, "x1": 2.0})
print(f"2D Sphere at (1.0, 2.0): {result}")

# 3D Sphere function
sphere_3d = SphereFunction(n_dim=3)
search_space = sphere_3d.search_space
result = sphere_3d({"x0": 1.0, "x1": 2.0, "x2": 3.0})
print(f"3D Sphere at (1.0, 2.0, 3.0): {result}")
