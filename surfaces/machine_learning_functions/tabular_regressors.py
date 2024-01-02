import os
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from .datasets import diabetes_data

from .base_machine_learning_function import MachineLearningFunction


class KNeighborsRegressorFunction(MachineLearningFunction):
    name = "KNeighbors Regressor Function"
    _name_ = "k_neighbors_regressor"
    __name__ = "KNeighborsRegressorFunction"

    def __init__(self, input_type="dictionary", sleep=0):
        super().__init__()

        self.input_type = input_type
        self.sleep = sleep

        self.search_space = {
            "n_neighbors": list(np.arange(3, 150, 5)),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "cv": [2, 3, 4, 5, 10],
            "dataset": [diabetes_data],
        }

    def objective_function(self, params):
        knc = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            algorithm=params["algorithm"],
        )
        X, y = params["dataset"]()
        scores = cross_val_score(knc, X, y, cv=params["cv"])
        return scores.mean()


class GradientBoostingRegressorFunction(MachineLearningFunction):
    name = "Gradient Boosting Regressor Function"
    _name_ = "gradient_boosting_regressor"
    __name__ = "GradientBoostingRegressorFunction"

    def __init__(self, input_type="dictionary", sleep=0):
        super().__init__()

        self.input_type = input_type
        self.sleep = sleep

        self.search_space = {
            "n_estimators": list(np.arange(5, 150, 5)),
            "max_depth": list(np.arange(1, 15)),
            "cv": [2, 3, 4, 5],
            "dataset": [diabetes_data],
        }

    def objective_function(self, params):
        knc = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
        X, y = params["dataset"]()
        scores = cross_val_score(knc, X, y, cv=params["cv"])
        return scores.mean()
