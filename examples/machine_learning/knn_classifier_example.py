"""KNN Classifier - Hyperparameter tuning for K-Nearest Neighbors."""

from surfaces.test_functions import KNeighborsClassifierFunction

knn = KNeighborsClassifierFunction()

# First evaluation
params1 = {"n_neighbors": 3, "algorithm": "auto"}
score1 = knn(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"n_neighbors": 10, "algorithm": "ball_tree"}
score2 = knn(params2)
print(f"params: {params2} -> score: {score2:.4f}")
