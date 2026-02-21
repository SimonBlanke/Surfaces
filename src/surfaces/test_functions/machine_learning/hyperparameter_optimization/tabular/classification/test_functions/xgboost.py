"""XGBoost Classifier test function with surrogate support."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from surfaces.modifiers import BaseModifier

from .._base_classification import BaseClassification
from ..datasets import DATASETS


class XGBoostClassifierFunction(BaseClassification):
    """XGBoost Classifier test function.

    Parameters
    ----------
    dataset : str, default="digits"
        Dataset to use.
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "XGBoost Classifier Function"
    _name_ = "xgboost_classifier"
    __name__ = "XGBoostClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = ["n_estimators", "max_depth", "learning_rate"]

    n_estimators_default = list(np.arange(50, 300, 25))
    max_depth_default = list(range(2, 12))
    learning_rate_default = [0.01, 0.05, 0.1, 0.2, 0.3]

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
            "max_depth": self.max_depth_default,
            "learning_rate": self.learning_rate_default,
        }

    def _create_objective_function(self) -> None:
        X, y = self._dataset_loader()
        cv = self.cv

        def objective(params: Dict[str, Any]) -> float:
            n_estimators = int(round(float(params["n_estimators"])))
            max_depth = int(round(float(params["max_depth"])))
            learning_rate = float(params["learning_rate"])

            clf = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
            )

            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            return float(scores.mean())

        self.pure_objective_function = objective
