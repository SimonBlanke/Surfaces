"""Regression tests for ML benchmark error handling."""

import importlib
import importlib.util

import numpy as np
import pytest

from surfaces.test_functions.machine_learning.hyperparameter_optimization.timeseries.forecasting.test_functions.exp_smoothing_forecaster import (
    ExpSmoothingForecasterFunction,
)


class _ValueErrorForecaster:
    def __init__(self, **kwargs):
        pass

    def fit(self, y_train):
        raise ValueError("invalid Holt-Winters configuration")


class _RuntimeErrorForecaster:
    def __init__(self, **kwargs):
        pass

    def fit(self, y_train):
        raise RuntimeError("unexpected implementation failure")


def test_exp_smoothing_expected_model_value_error_scores_zero(monkeypatch):
    exp_smoothing = pytest.importorskip("sktime.forecasting.exp_smoothing")
    monkeypatch.setattr(exp_smoothing, "ExponentialSmoothing", _ValueErrorForecaster)

    func = ExpSmoothingForecasterFunction()

    assert func({"trend": "add", "seasonal": "mul", "sp": 12}) == 0.0


def test_exp_smoothing_unexpected_model_error_is_not_silenced(monkeypatch):
    exp_smoothing = pytest.importorskip("sktime.forecasting.exp_smoothing")
    monkeypatch.setattr(exp_smoothing, "ExponentialSmoothing", _RuntimeErrorForecaster)

    func = ExpSmoothingForecasterFunction()

    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        func({"trend": "add", "seasonal": "mul", "sp": 12})


def _classification_data(n_samples=18, n_features=6, n_classes=3):
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    y = np.arange(n_samples) % n_classes
    return X, y


def _regression_data(n_samples=18, n_features=6):
    X = np.arange(n_samples * n_features, dtype=float).reshape(n_samples, n_features)
    y = np.linspace(0.0, 1.0, n_samples)
    return X, y


def _import_class(import_path):
    module_name, class_name = import_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _capture_cross_val_score(monkeypatch):
    model_selection = pytest.importorskip("sklearn.model_selection")
    calls = []

    def fake_cross_val_score(estimator, X, y, cv=None, scoring=None, error_score=None, **kwargs):
        calls.append(
            {
                "estimator": estimator,
                "X": X,
                "y": y,
                "cv": cv,
                "scoring": scoring,
                "error_score": error_score,
                "kwargs": kwargs,
            }
        )
        return np.array([0.25, 0.75])

    monkeypatch.setattr(model_selection, "cross_val_score", fake_cross_val_score)
    return calls


def _assert_single_pipeline_call(calls, expected_steps, expected_x_shape):
    pipeline_module = pytest.importorskip("sklearn.pipeline")

    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call["estimator"], pipeline_module.Pipeline)
    assert [name for name, _ in call["estimator"].steps] == expected_steps
    assert call["X"].shape == expected_x_shape
    assert call["error_score"] == "raise"


def test_polynomial_feature_transformation_uses_regression_dataset_names():
    pytest.importorskip("sklearn")

    from surfaces.test_functions.machine_learning import PolynomialFeatureTransformationFunction

    assert PolynomialFeatureTransformationFunction.available_datasets == ["diabetes", "california"]
    PolynomialFeatureTransformationFunction(dataset="california")

    with pytest.raises(ValueError, match="california_housing"):
        PolynomialFeatureTransformationFunction(dataset="california_housing")


def test_kneighbors_classifier_default_neighbors_fit_small_cv_training_folds():
    pytest.importorskip("sklearn")

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    func = KNeighborsClassifierFunction(dataset="iris", cv=5)
    max_neighbors = func._max_valid_n_neighbors(150)

    assert max(func.search_space["n_neighbors"]) <= max_neighbors


def test_kneighbors_classifier_rejects_neighbors_larger_than_cv_training_fold():
    pytest.importorskip("sklearn")

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    func = KNeighborsClassifierFunction(dataset="iris", cv=5)

    with pytest.raises(ValueError, match="supports at most 120 neighbors"):
        func({"n_neighbors": 121, "algorithm": "auto"})


def test_kneighbors_classifier_cross_validation_raises_model_errors(monkeypatch):
    pytest.importorskip("sklearn")

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    calls = _capture_cross_val_score(monkeypatch)
    func = KNeighborsClassifierFunction(dataset="iris", cv=5)

    result = func._ml_objective({"n_neighbors": 3, "algorithm": "auto"})

    assert result == 0.5
    assert calls[0]["error_score"] == "raise"


@pytest.mark.parametrize(
    (
        "import_path",
        "init_kwargs",
        "params",
        "expected_steps",
        "data_factory",
        "skip_package",
    ),
    [
        (
            "surfaces.test_functions.machine_learning.FeatureScalingPipelineFunction",
            {"dataset": "iris", "cv": 2},
            {"scaler": "standard", "model_type": "svm"},
            ["scaler", "model"],
            _classification_data,
            None,
        ),
        (
            "surfaces.test_functions.machine_learning.MutualInfoFeatureSelectionFunction",
            {"dataset": "iris", "cv": 2},
            {"n_features": 4, "model_type": "rf"},
            ["feature_selection", "model"],
            _classification_data,
            None,
        ),
        (
            "surfaces.test_functions.machine_learning.PolynomialFeatureTransformationFunction",
            {"dataset": "diabetes", "cv": 2},
            {"degree": 2, "interaction_only": False, "include_bias": True},
            ["polynomial_features", "model"],
            _regression_data,
            None,
        ),
        (
            "surfaces.test_functions.machine_learning.SVMImageClassifierFunction",
            {"dataset": "mnist", "cv": 2, "n_components": 2},
            {"C": 1.0, "kernel": "rbf", "gamma": "scale"},
            ["scaler", "pca", "model"],
            _classification_data,
            None,
        ),
        (
            "surfaces.test_functions.machine_learning.RandomForestImageClassifierFunction",
            {"dataset": "mnist", "cv": 2, "n_components": 2},
            {"n_estimators": 10, "max_depth": 5},
            ["scaler", "pca", "model"],
            _classification_data,
            None,
        ),
        (
            "surfaces.test_functions.machine_learning.XGBoostImageClassifierFunction",
            {"dataset": "mnist", "cv": 2, "n_components": 2},
            {"n_estimators": 10, "max_depth": 3, "learning_rate": 0.1},
            ["scaler", "pca", "model"],
            _classification_data,
            "xgboost",
        ),
        (
            "surfaces.test_functions.machine_learning.KNNTSClassifierFunction",
            {"dataset": "gunpoint", "cv": 2},
            {"n_neighbors": 3, "metric": "euclidean"},
            ["scaler", "model"],
            _classification_data,
            None,
        ),
    ],
)
def test_preprocessing_steps_are_fitted_inside_cross_validation(
    monkeypatch,
    import_path,
    init_kwargs,
    params,
    expected_steps,
    data_factory,
    skip_package,
):
    pytest.importorskip("sklearn")
    if skip_package is not None and importlib.util.find_spec(skip_package) is None:
        pytest.skip(f"{skip_package} is not installed")

    calls = _capture_cross_val_score(monkeypatch)
    func_class = _import_class(import_path)
    func = func_class(**init_kwargs)
    X, y = data_factory()
    monkeypatch.setattr(func, "_get_training_data", lambda: (X, y))

    result = func._ml_objective(params)

    assert result == 0.5
    _assert_single_pipeline_call(calls, expected_steps, X.shape)
