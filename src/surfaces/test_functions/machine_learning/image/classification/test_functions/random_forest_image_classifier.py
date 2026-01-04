"""Random Forest Image Classifier test function."""

from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .._base_image_classification import BaseImageClassification
from ..datasets import DATASETS


class RandomForestImageClassifierFunction(BaseImageClassification):
    """Random Forest Image Classifier test function.

    Uses Random Forest on PCA-reduced image features for classification.
    A sklearn-only approach without deep learning dependencies.

    Parameters
    ----------
    dataset : str, default="mnist"
        Dataset to use. One of: "mnist", "fashion_mnist".
    cv : int, default=3
        Number of cross-validation folds.
    n_components : int, default=50
        Number of PCA components to retain.
    use_surrogate : bool, default=False
        If True, use pre-trained surrogate for fast evaluation.
    objective : str, default="maximize"
        Either "minimize" or "maximize".
    sleep : float, default=0
        Artificial delay in seconds.

    Examples
    --------
    >>> from surfaces.test_functions.machine_learning.image import (
    ...     RandomForestImageClassifierFunction
    ... )
    >>> func = RandomForestImageClassifierFunction(dataset="mnist")
    >>> result = func({"n_estimators": 100, "max_depth": 10})
    """

    name = "Random Forest Image Classifier Function"
    _name_ = "random_forest_image_classifier"
    __name__ = "RandomForestImageClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5]

    # Search space parameters
    para_names = ["n_estimators", "max_depth"]
    n_estimators_default = list(np.arange(10, 200, 20))
    max_depth_default = [None] + list(np.arange(5, 30, 5))

    def __init__(
        self,
        dataset: str = "mnist",
        cv: int = 3,
        n_components: int = 50,
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
            raise ValueError(
                f"Unknown dataset '{dataset}'. " f"Available: {self.available_datasets}"
            )

        if cv not in self.available_cv:
            raise ValueError(f"Invalid cv={cv}. Available: {self.available_cv}")

        self.dataset = dataset
        self.cv = cv
        self.n_components = n_components
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
        """Search space containing hyperparameters."""
        return {
            "n_estimators": self.n_estimators_default,
            "max_depth": self.max_depth_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        X_raw, y = self._dataset_loader()

        # Apply PCA for dimensionality reduction
        scaler = StandardScaler()
        pca = PCA(n_components=self.n_components, random_state=42)
        X_scaled = scaler.fit_transform(X_raw)
        X = pca.fit_transform(X_scaled)

        cv = self.cv

        def random_forest_image_classifier(params: Dict[str, Any]) -> float:
            model = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = random_forest_image_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
            "n_components": self.n_components,
        }
