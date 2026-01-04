"""Random Forest Classifier - Hyperparameter tuning for ensemble models."""

from surfaces.test_functions import RandomForestClassifierFunction

rf = RandomForestClassifierFunction()

# First evaluation
params1 = {
    "n_estimators": 10,
    "max_depth": 3,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "criterion": "gini",
    "bootstrap": True,
}
score1 = rf(params1)
print(f"params: {params1} -> score: {score1:.4f}")

# Second evaluation
params2 = {
    "n_estimators": 50,
    "max_depth": 10,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "criterion": "entropy",
    "bootstrap": True,
}
score2 = rf(params2)
print(f"params: {params2} -> score: {score2:.4f}")
