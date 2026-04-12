"""CatBoost Classifier test function for tabular ML."""

from typing import Any, Dict, List, Optional

from surfaces.modifiers import BaseModifier

from .._base_classification import BaseClassification
from ..datasets import DATASETS


class CatBoostClassifierFunction(BaseClassification):
    """CatBoost Classifier test function.

    Parameters
    ----------
    dataset : str, default="digits"
        Dataset to use. One of: "digits", "iris", "wine", "breast_cancer", "covtype".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "CatBoost Classifier Function"
    _name_ = "catboost_classifier"
    _dependencies = {"ml": ["sklearn", "catboost"]}

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = [
        "iterations",
        "depth",
        "learning_rate",
        "l2_leaf_reg",
        "random_strength",
    ]

    iterations_default = list(range(50, 300, 25))
    depth_default = list(range(3, 11))
    learning_rate_default = [0.01, 0.03, 0.05, 0.1, 0.2]
    l2_leaf_reg_default = [1, 3, 5, 7, 9]
    random_strength_default = [0, 0.1, 0.5, 1.0, 2.0]

    latex_formula = (
        r"\text{CV-Accuracy} = f(\text{iterations}, \text{depth}, \text{learning\_rate}, \dots)"
    )
    tagline = (
        "Cross-validated accuracy of a CatBoost classifier. "
        "Gradient boosting with ordered boosting for categorical-friendly tree learning."
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

    def _default_search_space(self) -> Dict[str, Any]:
        return {
            "iterations": self.iterations_default,
            "depth": self.depth_default,
            "learning_rate": self.learning_rate_default,
            "l2_leaf_reg": self.l2_leaf_reg_default,
            "random_strength": self.random_strength_default,
        }

    def _ml_objective(self, params: Dict[str, Any]) -> float:
        from catboost import CatBoostClassifier
        from sklearn.model_selection import cross_val_score

        X, y = self._get_training_data()
        clf = CatBoostClassifier(
            iterations=params["iterations"],
            depth=params["depth"],
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            random_strength=params["random_strength"],
            random_seed=42,
            thread_count=-1,
            allow_writing_files=False,
            verbose=False,
        )
        scores = cross_val_score(clf, X, y, cv=self.cv, scoring="accuracy")
        return scores.mean()

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**params, "dataset": self.dataset, "cv": self.cv}
