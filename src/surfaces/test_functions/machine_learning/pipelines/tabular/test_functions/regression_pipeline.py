"""Regression Pipeline test function."""

from typing import Any, Callable, Dict, List, Optional, Union

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from surfaces.modifiers import BaseModifier
from surfaces.test_functions.machine_learning.hyperparameter_optimization.tabular.regression.datasets import (
    DATASETS,
)

from .._base_tabular_pipeline import BaseTabularPipeline


class RegressionPipelineFunction(BaseTabularPipeline):
    """Regression Pipeline test function.

    Optimizes an end-to-end regression pipeline including feature transformation,
    scaling, and model selection. Evaluates different combinations to find the
    best preprocessing and algorithm configuration.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use for evaluation. One of: "diabetes", "california".
    cv : int, default=5
        Number of cross-validation folds.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    modifiers : list of BaseModifier, optional
        List of modifiers to apply to function evaluations.

    Examples
    --------
    >>> from surfaces.test_functions import RegressionPipelineFunction
    >>> func = RegressionPipelineFunction(dataset="diabetes", cv=5)
    >>> func.search_space
    {'use_poly': [True, False], 'scaler': ['standard', 'minmax', 'none'], ...}
    >>> result = func({"use_poly": True, "scaler": "standard", "model_type": "ridge"})
    """

    name = "Regression Pipeline"
    _name_ = "regression_pipeline"
    __name__ = "RegressionPipelineFunction"

    available_datasets = ["diabetes", "california"]
    available_cv = [2, 3, 5, 10]

    para_names = ["use_poly", "scaler", "model_type"]
    use_poly_default = [True, False]
    scaler_default = ["standard", "minmax", "none"]
    model_type_default = ["ridge", "dt", "rf", "gb"]

    def __init__(
        self,
        dataset: str = "diabetes",
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
        """Search space for regression pipeline optimization."""
        return {
            "use_poly": self.use_poly_default,
            "scaler": self.scaler_default,
            "model_type": self.model_type_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function for regression pipeline."""
        X, y = self._dataset_loader()
        cv = self.cv

        def objective_function(params: Dict[str, Any]) -> float:
            # Build pipeline steps
            steps = []

            # 1. Polynomial features (optional)
            if params["use_poly"]:
                steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))

            # 2. Scaling step
            scaler_type = params["scaler"]
            if scaler_type == "standard":
                steps.append(("scaler", StandardScaler()))
            elif scaler_type == "minmax":
                steps.append(("scaler", MinMaxScaler()))
            # "none" means no scaling

            # 3. Model step
            model_type = params["model_type"]
            if model_type == "ridge":
                model = Ridge(alpha=1.0, random_state=42)
            elif model_type == "dt":
                model = DecisionTreeRegressor(max_depth=7, random_state=42)
            elif model_type == "rf":
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif model_type == "gb":
                model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            steps.append(("model", model))

            # Create and evaluate pipeline
            pipeline = Pipeline(steps)
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="r2")
            return scores.mean()

        self.pure_objective_function = objective_function
