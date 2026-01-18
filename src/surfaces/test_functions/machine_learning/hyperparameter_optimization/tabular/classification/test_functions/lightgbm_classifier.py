"""LightGBM Classifier test function with surrogate support."""

from typing import Any, Dict, List, Optional

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

from surfaces.modifiers import BaseModifier

from .._base_classification import BaseClassification
from ..datasets import DATASETS


class LightGBMClassifierFunction(BaseClassification):
    """LightGBM Classifier test function.

    Parameters
    ----------
    dataset : str, default="digits"
        Dataset to use. One of: "digits", "iris", "wine", "breast_cancer", "covtype".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "LightGBM Classifier Function"
    _name_ = "lightgbm_classifier"
    __name__ = "LightGBMClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = [
        "n_estimators",
        "learning_rate",
        "num_leaves",
        "max_depth",
        "min_child_samples",
        "subsample",
        "colsample_bytree",
        "reg_alpha",
        "reg_lambda",
    ]

    # Hp search space defaults

    n_estimators_default = list(np.arange(10, 300, 10))
    learning_rate_default = [1e-3, 1e-1, 0.5, 1.0]
    num_leaves_default = list(range(10, 100, 5))
    max_depth_default = list(range(2, 20, 1))
    min_child_samples_default = list(range(5, 100, 5))
    subsample_default = list(np.arange(0.1, 1.01, 0.1))
    colsample_bytree_default = list(np.arange(0.1, 1.01, 0.1))
    reg_alpha_default = [0, 0.001, 0.01, 0.1, 1, 10]
    reg_lambda_default = [0, 0.001, 0.01, 0.1, 10]

    # Function sheet for doc
    latex_formula = r"\text{CV-Accuracy} = f(\text{n\_estimators}, \text{learning\_rate}, \dots)"
    tagline = (
        "Cross-validated accuracy of a LightGBM classifier. "
        "Gradient boosting with tree-based learning."
    )

    def __init__(
        self,
        dataset: str = "digits",
        cv: int = 5,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks=None,
        catch_errors=None,
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
            modifiers=modifiers,
            memory=memory,
            collect_data=collect_data,
            callbacks=callbacks,
            catch_errors=catch_errors,
            use_surrogate=use_surrogate,
        )

    @property
    def search_space(self) -> Dict[str, Any]:
        return {
            "n_estimators": self.n_estimators_default,
            "learning_rate": self.learning_rate_default,
            "num_leaves": self.num_leaves_default,
            "max_depth": self.max_depth_default,
            "min_child_samples": self.min_child_samples_default,
            "subsample": self.subsample_default,
            "colsample_bytree": self.colsample_bytree_default,
            "reg_alpha": self.reg_alpha_default,
            "reg_lambda": self.reg_lambda_default,
        }

    def _create_objective_function(self) -> None:
        """
        Creates the objective function closure with fixed data
        """
        X, y = self._dataset_loader()
        cv = self.cv

        def objective(params: Dict[str, Any]) -> float:
            clf = LGBMClassifier(
                n_estimators=params["n_estimators"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                max_depth=params["max_depth"],
                min_child_samples=params["min_child_samples"],
                subsample=params["subsample"],
                colsample_bytree=params["colsample_bytree"],
                reg_alpha=params["reg_alpha"],
                reg_lambda=params["reg_lambda"],
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**params, "dataset": self.dataset, "cv": self.cv}
