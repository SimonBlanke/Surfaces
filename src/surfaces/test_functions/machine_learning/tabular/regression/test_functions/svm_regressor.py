"""Support Vector Machine Regressor test function with surrogate support."""

from typing import Any, Dict

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR

from .._base_regression import BaseRegression
from ..datasets import DATASETS


class SVMRegressorFunction(BaseRegression):
    """Support Vector Machine Regressor test function.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use. One of: "diabetes", "california", "friedman1", "friedman2", "linear".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "SVM Regressor Function"
    _name_ = "svm_regressor"
    __name__ = "SVMRegressorFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = ["C", "kernel", "gamma"]
    C_default = [0.01, 0.1, 1.0, 10.0, 100.0]
    kernel_default = ["linear", "rbf", "poly", "sigmoid"]
    gamma_default = ["scale", "auto"]

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
            "C": self.C_default,
            "kernel": self.kernel_default,
            "gamma": self.gamma_default,
        }

    def _create_objective_function(self) -> None:
        X, y = self._dataset_loader()
        cv = self.cv

        def objective(params: Dict[str, Any]) -> float:
            reg = SVR(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params["gamma"],
            )
            scores = cross_val_score(reg, X, y, cv=cv, scoring="r2")
            return scores.mean()

        self.pure_objective_function = objective

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**params, "dataset": self.dataset, "cv": self.cv}
