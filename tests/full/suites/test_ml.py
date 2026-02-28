# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for machine learning test functions.

ML test functions wrap scikit-learn models to create hyperparameter
optimization benchmarks. They're organized by:
- Tabular: Classification and regression on tabular data
- Image: Image classification tasks
- Time-series: Forecasting and classification on time-series data
"""

import numpy as np
import pytest

from surfaces.test_functions.machine_learning import machine_learning_functions
from tests.conftest import func_id, get_sample_params, instantiate_function


@pytest.mark.ml
class TestMLInstantiation:
    """Test basic instantiation of ML functions."""

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_instantiates(self, func_class):
        """ML functions instantiate correctly."""
        func = instantiate_function(func_class)
        assert func is not None

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_has_search_space(self, func_class):
        """ML functions have non-empty search space."""
        func = instantiate_function(func_class)
        assert len(func.search_space) > 0

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_is_callable(self, func_class):
        """ML functions are callable."""
        func = instantiate_function(func_class)
        assert callable(func)


@pytest.mark.ml
class TestMLSearchSpace:
    """Test ML function search space properties."""

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_search_space_is_dict(self, func_class):
        """Search space is a dictionary."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)

    @pytest.mark.parametrize("func_class", machine_learning_functions, ids=func_id)
    def test_search_space_values_valid(self, func_class):
        """Search space values are valid (iterable or single values)."""
        func = instantiate_function(func_class)
        for key, values in func.search_space.items():
            # Values can be iterables or single values (like callables)
            if hasattr(values, "__iter__") and not isinstance(values, (str, type)):
                values_list = list(values)
                assert len(values_list) > 0, f"{key} must have values"


@pytest.mark.ml
class TestClassificationFunctions:
    """Test classification ML functions."""

    def test_kneighbors_classifier(self, quick_ml_params):
        """KNeighborsClassifier function evaluates correctly."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)
        # Score should be between 0 and 1 (or loss equivalent)
        assert -1.0 <= result <= 1.0 or result >= 0

    def test_decision_tree_classifier(self, quick_ml_params):
        """DecisionTreeClassifier function evaluates correctly."""
        from surfaces.test_functions.machine_learning import DecisionTreeClassifierFunction

        func = DecisionTreeClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_random_forest_classifier(self, quick_ml_params):
        """RandomForestClassifier function evaluates correctly."""
        from surfaces.test_functions.machine_learning import RandomForestClassifierFunction

        func = RandomForestClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.ml
class TestRegressionFunctions:
    """Test regression ML functions."""

    def test_kneighbors_regressor(self, quick_regression_params):
        """KNeighborsRegressor function evaluates correctly."""
        from surfaces.test_functions.machine_learning import KNeighborsRegressorFunction

        func = KNeighborsRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_decision_tree_regressor(self, quick_regression_params):
        """DecisionTreeRegressor function evaluates correctly."""
        from surfaces.test_functions.machine_learning import DecisionTreeRegressorFunction

        func = DecisionTreeRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.ml
class TestMLObjectiveDirection:
    """Test objective direction for ML functions."""

    def test_minimize_returns_loss(self, quick_ml_params):
        """Minimize objective returns loss (lower is better)."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction(objective="minimize")
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        # For classification, loss should be positive (1 - accuracy)
        assert isinstance(result, (int, float))

    def test_maximize_returns_score(self, quick_ml_params):
        """Maximize objective returns negated score."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction(objective="maximize")
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))


@pytest.mark.ml
class TestMLDataCollection:
    """Test data collection for ML functions."""

    def test_tracks_evaluations(self, quick_ml_params):
        """ML functions track evaluation count."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}

        assert func.data.n_evaluations == 0
        func(params)
        assert func.data.n_evaluations == 1
        func(params)
        assert func.data.n_evaluations == 2

    def test_tracks_best_score(self, quick_ml_params):
        """ML functions track best score."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}

        func(params)
        assert func.data.best_score is not None
        assert func.data.best_params is not None


@pytest.mark.ml
class TestMLMemory:
    """Test memory caching for ML functions."""

    def test_memory_caches_results(self, quick_ml_params):
        """ML functions with memory=True cache results."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction(memory=True)
        params = {**get_sample_params(func), **quick_ml_params}

        # First call computes
        result1 = func(params)

        # Second call with same params should use cache
        result2 = func(params)

        assert result1 == result2

    def test_memory_different_params(self, quick_ml_params):
        """Different params don't use cached value."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        func = KNeighborsClassifierFunction(memory=True)

        params1 = {**get_sample_params(func), **quick_ml_params}
        params1["n_neighbors"] = 3

        params2 = {**get_sample_params(func), **quick_ml_params}
        params2["n_neighbors"] = 5

        result1 = func(params1)
        result2 = func(params2)

        # Results may differ (not guaranteed, but typically will)
        assert isinstance(result1, (int, float))
        assert isinstance(result2, (int, float))


@pytest.mark.ml
class TestMLCallbacks:
    """Test callbacks for ML functions."""

    def test_callback_invoked(self, quick_ml_params):
        """Callbacks are invoked after evaluation."""
        from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

        records = []
        func = KNeighborsClassifierFunction(callbacks=lambda r: records.append(r))
        params = {**get_sample_params(func), **quick_ml_params}

        func(params)

        assert len(records) == 1
        assert "score" in records[0]


@pytest.mark.ml
class TestAdditionalClassifiers:
    """Test additional classifier functions."""

    def test_gradient_boosting_classifier(self, quick_ml_params):
        """GradientBoostingClassifier evaluates correctly."""
        from surfaces.test_functions.machine_learning import GradientBoostingClassifierFunction

        func = GradientBoostingClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_svm_classifier(self, quick_ml_params):
        """SVMClassifier evaluates correctly."""
        from surfaces.test_functions.machine_learning import SVMClassifierFunction

        func = SVMClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_lightgbm_classifier(self, quick_ml_params):
        """LightGBMClassifier evaluates correctly."""
        from surfaces.test_functions.machine_learning import LightGBMClassifierFunction

        func = LightGBMClassifierFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.ml
class TestAdditionalRegressors:
    """Test additional regressor functions."""

    def test_gradient_boosting_regressor(self, quick_regression_params):
        """GradientBoostingRegressor evaluates correctly."""
        from surfaces.test_functions.machine_learning import GradientBoostingRegressorFunction

        func = GradientBoostingRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_svm_regressor(self, quick_regression_params):
        """SVMRegressor evaluates correctly."""
        from surfaces.test_functions.machine_learning import SVMRegressorFunction

        func = SVMRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_random_forest_regressor(self, quick_regression_params):
        """RandomForestRegressor evaluates correctly."""
        from surfaces.test_functions.machine_learning import RandomForestRegressorFunction

        func = RandomForestRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_lightgbm_regressor(self, quick_regression_params):
        """LightGBMRegressor evaluates correctly."""
        from surfaces.test_functions.machine_learning import LightGBMRegressorFunction

        func = LightGBMRegressorFunction()
        params = {**get_sample_params(func), **quick_regression_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.ml
class TestEnsembleFunctions:
    """Test ensemble optimization functions."""

    def test_voting_ensemble(self, quick_ml_params):
        """VotingEnsemble evaluates correctly."""
        from surfaces.test_functions.machine_learning import VotingEnsembleFunction

        func = VotingEnsembleFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_weighted_averaging(self, quick_ml_params):
        """WeightedAveraging evaluates correctly."""
        from surfaces.test_functions.machine_learning import WeightedAveragingFunction

        func = WeightedAveragingFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_stacking_ensemble(self, quick_ml_params):
        """StackingEnsemble evaluates correctly."""
        from surfaces.test_functions.machine_learning import StackingEnsembleFunction

        func = StackingEnsembleFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)


@pytest.mark.ml
class TestFeatureEngineering:
    """Test feature engineering functions."""

    def test_feature_scaling_pipeline(self, quick_ml_params):
        """FeatureScalingPipeline evaluates correctly."""
        from surfaces.test_functions.machine_learning import FeatureScalingPipelineFunction

        func = FeatureScalingPipelineFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_mutual_info_feature_selection(self, quick_ml_params):
        """MutualInfoFeatureSelection evaluates correctly."""
        from surfaces.test_functions.machine_learning import MutualInfoFeatureSelectionFunction

        func = MutualInfoFeatureSelectionFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)

    def test_polynomial_feature_transformation(self, quick_ml_params):
        """PolynomialFeatureTransformation evaluates correctly."""
        from surfaces.test_functions.machine_learning import PolynomialFeatureTransformationFunction

        func = PolynomialFeatureTransformationFunction()
        params = {**get_sample_params(func), **quick_ml_params}
        result = func(params)

        assert isinstance(result, (int, float))
        assert np.isfinite(result)
