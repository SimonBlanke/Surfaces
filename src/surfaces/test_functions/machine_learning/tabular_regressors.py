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

    para_names = ["n_neighbors", "algorithm", "cv", "dataset"]

    n_neighbors_default = list(np.arange(3, 150, 5))
    algorithm_default = ["auto", "ball_tree", "kd_tree", "brute"]
    cv_default = [2, 3, 4, 5, 8, 10]
    dataset_default = [diabetes_data]

    def __init__(
        self,
        metric=None,
        sleep=0,
        evaluate_from_data=False,
    ):
        super().__init__(metric, sleep, evaluate_from_data)

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


class GradientBoostingRegressorFunction(MachineLearningFunction):
    name = "Gradient Boosting Regressor Function"
    _name_ = "gradient_boosting_regressor"
    __name__ = "GradientBoostingRegressorFunction"

    para_names = ["n_estimators", "max_depth", "cv", "dataset"]

    n_estimators_default = list(np.arange(3, 150, 5))
    max_depth_default = list(np.arange(2, 25))
    cv_default = [2, 3, 4, 5, 8, 10]
    dataset_default = [diabetes_data]

    def __init__(
        self,
        metric=None,
        sleep=0,
        evaluate_from_data=False,
    ):
        super().__init__(metric, sleep, evaluate_from_data)

    def search_space(
        self,
        n_estimators: list = None,
        max_depth: list = None,
        cv: list = None,
        dataset: list = None,
    ):
        search_space: dict = {}

        search_space["n_estimators"] = (
            self.n_estimators_default if n_estimators is None else n_estimators
        )
        search_space["max_depth"] = (
            self.max_depth_default if max_depth is None else max_depth
        )
        search_space["cv"] = self.cv_default if cv is None else cv
        search_space["dataset"] = self.dataset_default if dataset is None else dataset

        return search_space

    def create_objective_function(self):
        def gradient_boosting_regressor(params):
            knc = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
            )
            X, y = params["dataset"]()
            scores = cross_val_score(knc, X, y, cv=params["cv"], scoring=self.metric)
            return scores.mean()

        self.pure_objective_function = gradient_boosting_regressor
