"""Example: Using machine learning test functions."""

from surfaces.test_functions import KNeighborsClassifierFunction

# Create an ML-based test function
k_neighbors = KNeighborsClassifierFunction()

# Get the hyperparameter search space
search_space = k_neighbors.search_space
print(f"Search space keys: {list(search_space.keys())}")

# Evaluate with specific hyperparameters
result = k_neighbors({"n_neighbors": 5, "algorithm": "auto"})
print(f"KNN accuracy with n_neighbors=5: {result}")
