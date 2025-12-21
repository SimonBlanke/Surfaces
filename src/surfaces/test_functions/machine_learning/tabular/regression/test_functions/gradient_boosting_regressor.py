# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from .._base_regression import BaseRegression
from ..datasets import diabetes_data


class GradientBoostingRegressorFunction(BaseRegression):
    """Gradient Boosting Regressor test function.

    A machine learning test function that evaluates Gradient Boosting
    regression with different hyperparameters using cross-validation.

    Parameters
    ----------
    metric : str, default="r2"
        Scoring metric for cross-validation.
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    para_names : list
        Names of the hyperparameters: n_estimators, max_depth, cv, dataset.
    n_estimators_default : list
        Default values for n_estimators parameter (3 to 150, step 5).
    max_depth_default : list
        Default values for max_depth parameter (2 to 24).
    cv_default : list
        Default cross-validation fold options: 2, 3, 4, 5, 8, 10.
    dataset_default : list
        Default datasets (diabetes).

    Examples
    --------
    >>> from surfaces.test_functions import GradientBoostingRegressorFunction
    >>> func = GradientBoostingRegressorFunction()
    >>> search_space = func.search_space
    >>> list(search_space.keys())
    ['n_estimators', 'max_depth', 'cv', 'dataset']
    """

    name = "Gradient Boosting Regressor Function"
    _name_ = "gradient_boosting_regressor"
    __name__ = "GradientBoostingRegressorFunction"

    para_names = ["n_estimators", "max_depth", "cv", "dataset"]

    n_estimators_default = list(np.arange(3, 150, 5))
    max_depth_default = list(np.arange(2, 25))
    cv_default = [2, 3, 4, 5, 8, 10]
    dataset_default = [diabetes_data]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _search_space(
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
        search_space["max_depth"] = self.max_depth_default if max_depth is None else max_depth
        search_space["cv"] = self.cv_default if cv is None else cv
        search_space["dataset"] = self.dataset_default if dataset is None else dataset

        return search_space

    def _create_objective_function(self):
        def gradient_boosting_regressor(params):
            knc = GradientBoostingRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
            )
            X, y = params["dataset"]()
            scores = cross_val_score(knc, X, y, cv=params["cv"], scoring="r2")
            return scores.mean()

        self.pure_objective_function = gradient_boosting_regressor
