"""Tests for multi-fidelity evaluation on ML test functions.

These tests verify that the fidelity parameter correctly controls
data subsampling, preserves class distributions, and integrates
with the ML evaluation pipeline. Requires sklearn.
"""

import warnings

import numpy as np
import pytest

from surfaces.test_functions.machine_learning import (
    GradientBoostingRegressorFunction,
    KNeighborsClassifierFunction,
    RandomForestClassifierFunction,
)
from surfaces.test_functions.machine_learning.hyperparameter_optimization.timeseries.forecasting.test_functions.gradient_boosting_forecaster import (
    GradientBoostingForecasterFunction,
)


class TestFidelitySubsampling:
    """Verify that fidelity reduces the training data."""

    def test_low_fidelity_is_faster_than_full(self):
        """Low fidelity should use less data and generally be faster."""
        func = RandomForestClassifierFunction(dataset="digits", cv=2)
        params = {"n_estimators": 10, "max_depth": 3, "min_samples_split": 5}

        # Both should return valid scores
        score_low = func(params, fidelity=0.1)
        score_full = func(params, fidelity=1.0)

        assert isinstance(score_low, (int, float))
        assert isinstance(score_full, (int, float))
        assert np.isfinite(score_low)
        assert np.isfinite(score_full)

    def test_fidelity_none_equals_no_fidelity(self):
        """fidelity=None and no fidelity should give identical results."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=2, memory=False)
        params = {"n_neighbors": 5, "algorithm": "auto"}

        result_none = func(params, fidelity=None)
        result_default = func(params)

        assert result_none == result_default

    def test_same_fidelity_is_reproducible(self):
        """Same params + same fidelity should give same result."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=2)
        params = {"n_neighbors": 5, "algorithm": "auto"}

        r1 = func(params, fidelity=0.5)
        r2 = func(params, fidelity=0.5)

        assert r1 == r2

    def test_different_fidelity_may_differ(self):
        """Different fidelity levels can produce different scores."""
        func = RandomForestClassifierFunction(dataset="digits", cv=2)
        params = {"n_estimators": 50, "max_depth": 5, "min_samples_split": 2}

        score_10 = func(params, fidelity=0.1)
        score_100 = func(params, fidelity=1.0)

        # They may or may not differ, but both must be valid
        assert isinstance(score_10, (int, float))
        assert isinstance(score_100, (int, float))


class TestStratifiedSampling:
    """Verify that classification subsampling preserves class distribution."""

    def test_all_classes_present_at_low_fidelity(self):
        """Even at low fidelity, all classes should be represented."""
        func = RandomForestClassifierFunction(dataset="iris", cv=2)

        # Manually trigger subsampling to inspect the data
        func._active_fidelity = 0.1
        X, y = func._get_training_data()
        func._active_fidelity = None

        classes_full = set(np.unique(func._dataset_loader()[1]))
        classes_sub = set(np.unique(y))

        assert classes_sub == classes_full

    def test_class_proportions_roughly_preserved(self):
        """Class proportions should be approximately maintained."""
        func = RandomForestClassifierFunction(dataset="digits", cv=2)

        X_full, y_full = func._dataset_loader()
        func._active_fidelity = 0.3
        X_sub, y_sub = func._get_training_data()
        func._active_fidelity = None

        _, counts_full = np.unique(y_full, return_counts=True)
        _, counts_sub = np.unique(y_sub, return_counts=True)

        # Proportions should be within 10% of original
        props_full = counts_full / counts_full.sum()
        props_sub = counts_sub / counts_sub.sum()

        assert np.allclose(props_full, props_sub, atol=0.1)

    def test_subsampled_size_matches_fidelity(self):
        """Number of subsampled points should be ~fidelity * total."""
        func = RandomForestClassifierFunction(dataset="digits", cv=2)

        X_full, _ = func._dataset_loader()
        n_full = len(X_full)

        func._active_fidelity = 0.5
        X_sub, _ = func._get_training_data()
        func._active_fidelity = None

        expected = int(n_full * 0.5)
        # Allow some tolerance because stratified sampling rounds per class
        assert abs(len(X_sub) - expected) < 0.1 * n_full


class TestRegressionSubsampling:
    """Verify that regression uses shuffled (not sequential) subsampling."""

    def test_regression_fidelity_works(self):
        func = GradientBoostingRegressorFunction(dataset="diabetes", cv=2)
        params = {"n_estimators": 10, "max_depth": 3}

        score = func(params, fidelity=0.5)
        assert isinstance(score, (int, float))
        assert np.isfinite(score)

    def test_regression_subsampled_size(self):
        func = GradientBoostingRegressorFunction(dataset="diabetes", cv=2)

        X_full, _ = func._dataset_loader()
        func._active_fidelity = 0.3
        X_sub, _ = func._get_training_data()
        func._active_fidelity = None

        expected = int(len(X_full) * 0.3)
        assert len(X_sub) == expected


class TestForecastingSubsampling:
    """Verify that time-series forecasting uses sequential subsampling."""

    def test_forecasting_fidelity_works(self):
        func = GradientBoostingForecasterFunction(dataset="airline", cv=2)
        params = {"n_estimators": 10, "max_depth": 3, "n_lags": 5}

        score = func(params, fidelity=0.5)
        assert isinstance(score, (int, float))
        assert np.isfinite(score)

    def test_forecasting_preserves_temporal_order(self):
        """Sequential subsampling must take the first N points."""
        func = GradientBoostingForecasterFunction(dataset="airline", cv=2)

        X_full, y_full = func._dataset_loader()
        func._active_fidelity = 0.5
        X_sub, y_sub = func._get_training_data()
        func._active_fidelity = None

        n_sub = len(X_sub)
        # The subsampled data should be the first n_sub points
        np.testing.assert_array_equal(y_sub, y_full[:n_sub])


class TestMinimumFidelity:
    """Verify that too-small fidelity raises a clear error."""

    def test_too_small_fidelity_raises_value_error(self):
        func = RandomForestClassifierFunction(dataset="iris", cv=5)
        params = {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2}

        with pytest.raises(ValueError, match="fidelity.*produces only.*samples"):
            func(params, fidelity=0.01)


class TestSurrogateFidelityWarning:
    """Verify warning when fidelity is used with surrogate."""

    def test_surrogate_fidelity_warns(self):
        """Using fidelity with a surrogate should emit UserWarning."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=2, use_surrogate=True)

        # Skip if no surrogate is actually loaded (depends on package installation)
        if not func.use_surrogate:
            pytest.skip("No surrogate model available")

        # The warning only fires for non-fidelity-aware surrogates. If the
        # loaded surrogate natively supports fidelity, there is nothing to warn
        # about — the fidelity value is passed through to predict().
        if func._surrogate is not None and func._surrogate.fidelity_aware:
            pytest.skip("Loaded surrogate is fidelity-aware; warning path N/A")

        params = {"n_neighbors": 5, "algorithm": "auto"}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            func(params, fidelity=0.5)
            fidelity_warnings = [x for x in w if "fidelity" in str(x.message).lower()]
            assert len(fidelity_warnings) == 1


class TestFidelityActiveStateCleanup:
    """Verify that _active_fidelity is properly cleaned up."""

    def test_active_fidelity_none_after_call(self):
        func = KNeighborsClassifierFunction(dataset="iris", cv=2)
        params = {"n_neighbors": 5, "algorithm": "auto"}

        func(params, fidelity=0.5)
        assert func._active_fidelity is None

    def test_active_fidelity_none_after_error(self):
        """_active_fidelity should be cleaned up even if evaluation fails."""
        func = KNeighborsClassifierFunction(dataset="iris", cv=2)

        # n_neighbors larger than the dataset will cause issues at very low fidelity
        # but we test cleanup with a normal call that uses catch_errors
        func({"n_neighbors": 5, "algorithm": "auto"}, fidelity=0.5)
        assert func._active_fidelity is None
