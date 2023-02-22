import os
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score
from ..datasets import diabetes_data

from .base_machine_learning_function import BaseMachineLearningFunction


class GradientBoostingRegressorFunction(BaseMachineLearningFunction):
    __name__ = "gradient_boosting_regressor"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)

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
