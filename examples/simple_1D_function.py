"""Example: Using a simple 1D test function."""

from surfaces.test_functions import GramacyAndLeeFunction

# Create the 1D Gramacy & Lee function
gramacy_lee = GramacyAndLeeFunction()

# Get the search space
search_space = gramacy_lee.search_space
print(f"Search space: {list(search_space.keys())}")

# Evaluate at a point
result = gramacy_lee({"x0": 0.5})
print(f"Gramacy & Lee at x0=0.5: {result}")

# Get the global optimum
print(f"Global optimum x: {gramacy_lee.x_global}")
print(f"Global optimum f: {gramacy_lee.f_global}")
