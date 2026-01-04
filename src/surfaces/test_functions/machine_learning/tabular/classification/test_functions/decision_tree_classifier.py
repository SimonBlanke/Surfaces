from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Decision Tree Classifier test function with surrogate support."""

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .._base_classification import BaseClassification
from ..datasets import DATASETS


class DecisionTreeClassifierFunction(BaseClassification):
    """Decision Tree Classifier test function.

    Parameters
    ----------
    dataset : str, default="digits"
        Dataset to use. One of: "digits", "iris", "wine", "breast_cancer", "covtype".
    cv : int, default=5
        Number of cross-validation folds.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    """

    name = "Decision Tree Classifier Function"
    _name_ = "decision_tree_classifier"
    __name__ = "DecisionTreeClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    para_names = ["max_depth", "min_samples_split", "min_samples_leaf"]
    max_depth_default = [None] + list(range(2, 30))
    min_samples_split_default = [2, 5, 10, 20, 50]
    min_samples_leaf_default = [1, 2, 5, 10, 20]

    # Function sheet attributes
    latex_formula = r"\text{CV-Accuracy} = f(\text{max\_depth}, \text{min\_samples\_split}, \text{min\_samples\_leaf})"
    tagline = (
        "Cross-validated accuracy of a Decision Tree classifier. "
        "Tree depth and leaf constraints define the hyperparameter landscape."
    )
    display_bounds = {"max_depth": (2, 20), "min_samples_split": (2, 50)}
    display_projection = {"dims": ("max_depth", "min_samples_split"), "fixed": {"min_samples_leaf": 1}}
    reference = None
    reference_url = "https://scikit-learn.org/stable/modules/tree.html"

    def __init__(
        self,
        dataset: str = "digits",
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
            clf = DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                random_state=42,
            )
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {**params, "dataset": self.dataset, "cv": self.cv}
