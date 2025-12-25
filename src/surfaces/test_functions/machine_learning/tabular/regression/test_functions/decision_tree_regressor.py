from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Decision Tree Regressor test function with surrogate support."""

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from .._base_regression import BaseRegression
from ..datasets import DATASETS


class DecisionTreeRegressorFunction(BaseRegression):
    """Decision Tree Regressor test function.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use. One of: "diabetes", "california", "friedman1", "friedman2", "linear".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "Decision Tree Regressor Function"
    _name_ = "decision_tree_regressor"
    __name__ = "DecisionTreeRegressorFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = ["max_depth", "min_samples_split", "min_samples_leaf"]
    max_depth_default = [None] + list(range(2, 30))
    min_samples_split_default = [2, 5, 10, 20, 50]
    min_samples_leaf_default = [1, 2, 5, 10, 20]

    def __init__(
        self,
        dataset: str = "diabetes",
        cv: int = 5,
        objective: str = "maximize",
        sleep: float = 0,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
        noise=None,
        use_surrogate: bool = False,
    ):
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'. Available: {self.available_datasets}")
        if cv not in self.available_cv:
            raise ValueError(f"Invalid cv={cv}. Available: {self.available_cv}")

        self.dataset = dataset
        self.cv = cv
        self._dataset_loader = DATASETS[dataset]

        super().__init__(
            objective=objective,
            sleep=sleep,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            noise=noise,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        return {
            "max_depth": self.max_depth_default,
            "min_samples_split": self.min_samples_split_default,
            "min_samples_leaf": self.min_samples_leaf_default,
        }

    def _create_objective_function(self) -> None:
        X, y = self._dataset_loader()
        cv = self.cv

        def objective(params: Dict[str, Any]) -> float:
            reg = DecisionTreeRegressor(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42,
            )
            scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
            return scores.mean()

        self.pure_objective_function = objective

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**params, "dataset": self.dataset, "cv": self.cv}
