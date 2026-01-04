"""KNN Classifier - Hyperparameter tuning for K-Nearest Neighbors."""

from surfaces.test_functions import KNeighborsClassifierFunction

# Create the KNN test function
knn = KNeighborsClassifierFunction()

# Evaluate with specific hyperparameters
accuracy = knn({"n_neighbors": 5, "algorithm": "auto"})
print(f"KNN accuracy: {accuracy:.4f}")
