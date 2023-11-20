import os
import numpy as np

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from ..datasets import diabetes_data

from .base_machine_learning_function import BaseMachineLearningFunction


class KNeighborsRegressorFunction(BaseMachineLearningFunction):
    __name__ = "k_neighbors_classifier"

    def __init__(self, input_type="dictionary", sleep=0):
        super().__init__(input_type, sleep)

        self.search_space = {
            "n_neighbors": list(np.arange(3, 150)),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "cv": [2, 3, 4, 5, 10],
            "dataset": [diabetes_data],
        }

    def model(self, params):
        knc = KNeighborsRegressor(
            n_neighbors=params["n_neighbors"],
            algorithm=params["algorithm"],
        )
        X, y = params["dataset"]()
        scores = cross_val_score(knc, X, y, cv=params["cv"])
        return scores.mean()


class GradientBoostingRegressorFunction(BaseMachineLearningFunction):
    __name__ = "gradient_boosting_regressor"

    def __init__(self, input_type="dictionary", sleep=0):
        super().__init__(input_type, sleep)

        self.search_space = {
            "n_estimators": list(np.arange(5, 150)),
            "max_depth": list(np.arange(1, 15)),
            "cv": [2, 3, 4, 5],
            "dataset": [diabetes_data],
        }

    def model(self, params):
        knc = GradientBoostingRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
        )
        X, y = params["dataset"]()
        scores = cross_val_score(knc, X, y, cv=params["cv"])
        return scores.mean()