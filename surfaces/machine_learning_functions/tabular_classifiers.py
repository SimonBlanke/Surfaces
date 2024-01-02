import os
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
from .datasets import digits_data, wine_data, iris_data

from .base_machine_learning_function import MachineLearningFunction


class KNeighborsClassifierFunction(MachineLearningFunction):
    name = "KNeighbors Classifier Function"
    _name_ = "k_neighbors_classifier"
    __name__ = "KNeighborsClassifierFunction"

    def __init__(self, input_type="dictionary", sleep=0):
        super().__init__()

        self.input_type = input_type
        self.sleep = sleep

        self.search_space = {
            "n_neighbors": list(np.arange(3, 150, 5)),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "cv": [2, 3, 4, 5, 8, 10],
            "dataset": [digits_data, wine_data, iris_data],
        }

    def objective_function(self, params):
        knc = KNeighborsClassifier(
            n_neighbors=params["n_neighbors"],
            algorithm=params["algorithm"],
        )
        X, y = params["dataset"]()
        scores = cross_val_score(knc, X, y, cv=params["cv"])
        return scores.mean()
