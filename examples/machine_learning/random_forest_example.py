"""Random Forest Classifier - Hyperparameter tuning for ensemble models."""

from surfaces.test_functions import RandomForestClassifierFunction

# Create the Random Forest test function
rf = RandomForestClassifierFunction()

# Evaluate with specific hyperparameters
params = {"n_estimators": 10, "max_depth": 5, "min_samples_split": 2,
          "min_samples_leaf": 1, "criterion": "gini", "bootstrap": True}
accuracy = rf(params)
print(f"Random Forest accuracy: {accuracy:.4f}")
