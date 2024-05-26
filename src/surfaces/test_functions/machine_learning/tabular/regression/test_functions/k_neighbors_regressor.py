# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np

from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import cross_val_score
from ..datasets import diabetes_data

from .._base_regression import BaseRegression


class KNeighborsRegressorFunction(BaseRegression):
    name = "KNeighbors Regressor Function"
    _name_ = "k_neighbors_regressor"
    __name__ = "KNeighborsRegressorFunction"

    para_names = ["n_neighbors", "algorithm", "cv", "dataset"]

    n_neighbors_default = list(np.arange(3, 150, 5))
    algorithm_default = ["auto", "ball_tree", "kd_tree", "brute"]
    cv_default = [2, 3, 4, 5, 8, 10]
    dataset_default = [diabetes_data]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def search_space(
        self,
        n_neighbors: list = None,
        algorithm: list = None,
        cv: list = None,
        dataset: list = None,
    ):
        search_space: dict = {}

        search_space["n_neighbors"] = (
            self.n_neighbors_default if n_neighbors is None else n_neighbors
        )
        search_space["algorithm"] = (
            self.algorithm_default if algorithm is None else algorithm
        )
        search_space["cv"] = self.cv_default if cv is None else cv
        search_space["dataset"] = self.dataset_default if dataset is None else dataset

        return search_space

    def create_objective_function(self):
        def k_neighbors_regressor(params):
            knc = KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                algorithm=params["algorithm"],
            )
            X, y = params["dataset"]()
            scores = cross_val_score(knc, X, y, cv=params["cv"], scoring=self.metric)
            return scores.mean()

        self.pure_objective_function = k_neighbors_regressor
