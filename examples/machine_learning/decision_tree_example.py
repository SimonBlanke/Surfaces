"""Decision Tree Classifier - Hyperparameter tuning for Decision Trees."""

from surfaces.test_functions import DecisionTreeClassifierFunction

# Create the Decision Tree test function
dt = DecisionTreeClassifierFunction()

# Evaluate with specific hyperparameters
params = {"max_depth": 5, "min_samples_split": 2, "min_samples_leaf": 1, "criterion": "gini"}
accuracy = dt(params)
print(f"Decision Tree accuracy: {accuracy:.4f}")
