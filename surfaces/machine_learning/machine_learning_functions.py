import os
import numpy as np

from simple_data_collector import DataCollector

from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris

from .base_machine_learning_function import BaseMachineLearningFunction


iris_dataset = load_iris()
X, y = iris_dataset.data, iris_dataset.target


class KNeighborsClassifier(BaseMachineLearningFunction):
    __name__ = "k_neighbors_classifier"

    def __init__(self, metric="score", input_type="dictionary", sleep=0):
        super().__init__(metric, input_type, sleep)

        self.search_space = {
            "n_neighbors": np.arange(3, 10),
            "leaf_size": np.arange(3, 5),
        }

    def model(self, params):
        knc = KNeighborsClassifier_(
            n_neighbors=params["n_neighbors"],
            leaf_size=params["leaf_size"],
        )
        scores = cross_val_score(knc, X, y, cv=2)
        return scores.mean()
