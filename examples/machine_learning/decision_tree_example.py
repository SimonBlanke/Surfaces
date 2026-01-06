"""Decision Tree Classifier - Hyperparameter tuning for Decision Trees."""

from surfaces.test_functions.machine_learning import DecisionTreeClassifierFunction

dt = DecisionTreeClassifierFunction()

# First evaluation
params1 = {"max_depth": 3, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
score1 = dt(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {"max_depth": 10, "min_samples_split": 5, "min_samples_leaf": 2, "criterion": "entropy"}
score2 = dt(params2)
print(f"params: {params2} -> score: {score2:.4f}")
