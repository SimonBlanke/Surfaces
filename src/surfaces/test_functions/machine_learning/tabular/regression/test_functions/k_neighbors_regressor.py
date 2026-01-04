"""K-Nearest Neighbors Regressor test function with surrogate support."""

from typing import Any, Dict

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from .._base_regression import BaseRegression
from ..datasets import DATASETS


class KNeighborsRegressorFunction(BaseRegression):
    """K-Nearest Neighbors Regressor test function.

    A machine learning test function that evaluates K-Nearest Neighbors
    regression with different hyperparameters using cross-validation.

    Parameters
    ----------
    dataset : str, default="diabetes"
        Dataset to use for evaluation. One of: "diabetes", "california".
        This is a fixed parameter (like a coefficient), not part of the search space.
    cv : int, default=5
        Number of cross-validation folds.
        This is a fixed parameter, not part of the search space.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate model for fast evaluation (~1ms).
        Falls back to real evaluation if no surrogate is available.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds added to each evaluation.

    Attributes
    ----------
    available_datasets : list
        Available dataset names: ["diabetes", "california"].
    available_cv : list
        Available CV fold options: [2, 3, 5, 10].

    Examples
    --------
    Basic usage with real evaluation:

    >>> from surfaces.test_functions import KNeighborsRegressorFunction
    >>> func = KNeighborsRegressorFunction(dataset="diabetes", cv=5)
    >>> func.search_space
    {'n_neighbors': [3, 8, 13, ...], 'algorithm': ['auto', 'ball_tree', ...]}
    >>> result = func({"n_neighbors": 5, "algorithm": "auto"})

    Fast evaluation with surrogate (requires surfaces[surrogates]):

    >>> func = KNeighborsRegressorFunction(dataset="diabetes", cv=5, use_surrogate=True)
    >>> result = func({"n_neighbors": 5, "algorithm": "auto"})  # ~1ms
    """

    name = "KNeighbors Regressor Function"
    _name_ = "k_neighbors_regressor"
    __name__ = "KNeighborsRegressorFunction"

    # Available options (for validation and documentation)
    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5, 10]

    # Search space parameters (only actual hyperparameters)
    para_names = ["n_neighbors", "algorithm"]
    n_neighbors_default = list(np.arange(3, 150, 5))
    algorithm_default = ["auto", "ball_tree", "kd_tree", "brute"]

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
        # Validate dataset
        if dataset not in DATASETS:
            raise ValueError(
                f"Unknown dataset '{dataset}'. " f"Available: {self.available_datasets}"
            )

        # Validate cv
        if cv not in self.available_cv:
            raise ValueError(f"Invalid cv={cv}. Available: {self.available_cv}")

        # Store fixed parameters (like coefficients in math functions)
        self.dataset = dataset
        self.cv = cv

        # Load dataset for real evaluation
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
        """Search space containing only hyperparameters (not dataset/cv)."""
        return {
            "n_neighbors": self.n_neighbors_default,
            "algorithm": self.algorithm_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        # Load dataset once
        X, y = self._dataset_loader()
        cv = self.cv

        def k_neighbors_regressor(params: Dict[str, Any]) -> float:
            knr = KNeighborsRegressor(
                n_neighbors=params["n_neighbors"],
                algorithm=params["algorithm"],
            )
            scores = cross_val_score(knr, X, y, cv=cv, scoring="r2")
            return scores.mean()

        self.pure_objective_function = k_neighbors_regressor

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters (dataset, cv) to params for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
        }
