"""Classification Pipeline test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.classification.datasets import (
    DATASETS,
)

from .._base_tabular_pipeline import BaseTabularPipeline


class ClassificationPipelineFunction(BaseTabularPipeline):
    """Classification Pipeline test function.

    Optimizes an end-to-end classification pipeline including preprocessing
    (scaling) and model selection. This represents a simplified AutoML scenario
    where both data transformation and algorithm choice are optimized together.

    Parameters
    ----------
    dataset : str, default="wine"
        Dataset to use for evaluation. One of: "digits", "iris", "wine", "breast_cancer".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import ClassificationPipelineFunction
    >>> func = ClassificationPipelineFunction(dataset="wine", cv=5)
    >>> func.search_space
    {'scaler': ['standard', 'minmax', 'none'], 'model_type': ['dt', 'rf', ...]}
    >>> result = func({"scaler": "standard", "model_type": "rf", "model_param": "medium"})
    """

    name = "Classification Pipeline"
    _name_ = "classification_pipeline"
    __name__ = "ClassificationPipelineFunction"

    available_datasets = ["digits", "iris", "wine", "breast_cancer"]
    available_cv = [2, 3, 5, 10]

    para_names = ["scaler", "model_type", "model_param"]
    scaler_default = ["standard", "minmax", "none"]
    model_type_default = ["dt", "rf", "gb", "svm"]
    model_param_default = ["small", "medium", "large"]

    def __init__(
        self,
        dataset: str = "wine",
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
        """Search space for classification pipeline optimization."""
        return {
            "scaler": self.scaler_default,
            "model_type": self.model_type_default,
            "model_param": self.model_param_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for classification pipeline."""
        X, y = self._dataset_loader()
        cv = self.cv

        def objective_function(params: Dict[str, Any]) -> float:
            # Build pipeline steps
            steps = []

            # 1. Scaling step
            scaler_type = params["scaler"]
            if scaler_type == "standard":
                steps.append(("scaler", StandardScaler()))
            elif scaler_type == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            # "none" means no scaling

            # 2. Model step (with model-specific hyperparameters)
            model_type = params["model_type"]
            model_param = params["model_param"]

            # Map model_param to actual hyperparameters
            if model_type == "dt":
                depth_map = {"small": 3, "medium": 7, "large": 15}
                model = DecisionTreeClassifier(max_depth=depth_map[model_param], random_state=42)
            elif model_type == "rf":
                n_est_map = {"small": 20, "medium": 50, "large": 100}
                model = RandomForestClassifier(n_estimators=n_est_map[model_param], random_state=42)
            elif model_type == "gb":
                n_est_map = {"small": 20, "medium": 50, "large": 100}
                model = GradientBoostingClassifier(
                    n_estimators=n_est_map[model_param], random_state=42
                )
            elif model_type == "svm":
                c_map = {"small": 0.1, "medium": 1.0, "large": 10.0}
                model = SVC(C=c_map[model_param], random_state=42)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            steps.append(("model", model))

            # Create and evaluate pipeline
            pipeline = Pipeline(steps)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = objective_function
