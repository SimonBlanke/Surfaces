"""Stacking Ensemble test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
    DATASETS,
)

from .._base_tabular_ensemble import BaseTabularEnsemble


class StackingEnsembleFunction(BaseTabularEnsemble):
    """Stacking Ensemble test function.

    Optimizes a stacking ensemble by selecting base learners and the
    meta-learner (final estimator). Stacking combines predictions from
    multiple models using a meta-model to learn the optimal combination.

    Parameters
    ----------
    dataset : str, default="iris"
        Dataset to use for evaluation. One of: "digits", "iris", "wine", "breast_cancer".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import StackingEnsembleFunction
    >>> func = StackingEnsembleFunction(dataset="iris", cv=5)
    >>> func.search_space
    {'use_dt': [True, False], 'use_rf': [True, False], ...}
    >>> result = func({"use_dt": True, "use_rf": True, "use_gb": True,
    ...                "use_svm": False, "final_estimator": "lr"})
    """

    name = "Stacking Ensemble"
    _name_ = "stacking_ensemble"
    __name__ = "StackingEnsembleFunction"

    available_datasets = ["digits", "iris", "wine", "breast_cancer"]
    available_cv = [2, 3, 5, 10]

    para_names = ["use_dt", "use_rf", "use_gb", "use_svm", "final_estimator"]
    use_dt_default = [True, False]
    use_rf_default = [True, False]
    use_gb_default = [True, False]
    use_svm_default = [True, False]
    final_estimator_default = ["lr", "rf", "gb"]

    def __init__(
        self,
        dataset: str = "iris",
        cv: int = 5,
        objective: str = "maximize",
        modifiers: Optional[List[BaseModifier]] = None,
        memory: bool = False,
        collect_data: bool = True,
        callbacks: Optional[Union[Callable, List[Callable]]] = None,
        catch_errors: Optional[Dict[type, float]] = None,
        use_surrogate: bool = False,
    ):
        if dataset not in self.available_datasets:
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
        """Search space for stacking ensemble optimization."""
        return {
            "use_dt": self.use_dt_default,
            "use_rf": self.use_rf_default,
            "use_gb": self.use_gb_default,
            "use_svm": self.use_svm_default,
            "final_estimator": self.final_estimator_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for stacking ensemble."""
        X, y = self._dataset_loader()
        cv = self.cv

        def objective_function(params: Dict[str, Any]) -> float:
            # Build base estimators
            estimators = []

            if params["use_dt"]:
                estimators.append(("dt", DecisionTreeClassifier(random_state=42)))

            if params["use_rf"]:
                estimators.append(("rf", RandomForestClassifier(n_estimators=50, random_state=42)))

            if params["use_gb"]:
                estimators.append(
                    ("gb", GradientBoostingClassifier(n_estimators=50, random_state=42))
                )

            if params["use_svm"]:
                estimators.append(("svm", SVC(probability=True, random_state=42)))

            # Need at least 2 base models for stacking
            if len(estimators) < 2:
                return 0.0

            # Select final estimator (meta-learner)
            final_est_type = params["final_estimator"]
            if final_est_type == "lr":
                final_estimator = LogisticRegression(max_iter=1000, random_state=42)
            elif final_est_type == "rf":
                final_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            elif final_est_type == "gb":
                final_estimator = GradientBoostingClassifier(n_estimators=50, random_state=42)
            else:
                raise ValueError(f"Unknown final_estimator: {final_est_type}")

            # Create stacking classifier
            ensemble = StackingClassifier(
                estimators=estimators, final_estimator=final_estimator, cv=3
            )

            # Evaluate
            scores = cross_val_score(ensemble, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective_function
