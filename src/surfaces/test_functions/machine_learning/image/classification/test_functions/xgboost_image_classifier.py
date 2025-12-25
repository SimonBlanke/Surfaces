from typing import Any, Dict

# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""XGBoost Image Classifier test function."""

import numpy as np

from .._base_image_classification import BaseImageClassification
from ..datasets import DATASETS


def _check_xgboost():
    """Check if xgboost is available."""
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        raise ImportError(
            "XGBoost image classifier requires xgboost. "
            "Install with: pip install surfaces[xgboost]"
        )


class XGBoostImageClassifierFunction(BaseImageClassification):
    """XGBoost Image Classifier test function.

    Uses XGBoost on PCA-reduced image features for classification.
    XGBoost is a gradient boosting library optimized for speed and performance.

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
    ...     XGBoostImageClassifierFunction
    ... )
    >>> func = XGBoostImageClassifierFunction(dataset="mnist")
    >>> result = func({"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1})
    """

    name = "XGBoost Image Classifier Function"
    _name_ = "xgboost_image_classifier"
    __name__ = "XGBoostImageClassifierFunction"

    available_datasets = list(DATASETS.keys())
    available_cv = [2, 3, 5]

    # Search space parameters
    para_names = ["n_estimators", "max_depth", "learning_rate"]
    n_estimators_default = [50, 100, 150, 200, 250]
    max_depth_default = [3, 4, 5, 6, 7, 8, 10]
    learning_rate_default = [0.01, 0.05, 0.1, 0.2, 0.3]

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
        _check_xgboost()

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
            "learning_rate": self.learning_rate_default,
        }

    def _create_objective_function(self) -> None:
        """Create objective function with fixed dataset and cv."""
        from sklearn.decomposition import PCA
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier

        X_raw, y = self._dataset_loader()

        # Apply PCA for dimensionality reduction
        scaler = StandardScaler()
        pca = PCA(n_components=self.n_components, random_state=42)
        X_scaled = scaler.fit_transform(X_raw)
        X = pca.fit_transform(X_scaled)

        cv = self.cv
        n_classes = len(np.unique(y))

        def xgboost_image_classifier(params: Dict[str, Any]) -> float:
            model = XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                objective="multi:softmax" if n_classes > 2 else "binary:logistic",
                num_class=n_classes if n_classes > 2 else None,
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )
            scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            return scores.mean()

        self.pure_objective_function = xgboost_image_classifier

    def _get_surrogate_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Add fixed parameters for surrogate prediction."""
        return {
            **params,
            "dataset": self.dataset,
            "cv": self.cv,
            "n_components": self.n_components,
        }
