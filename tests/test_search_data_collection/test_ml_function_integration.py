# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import tempfile
import time
from unittest.mock import patch

import pytest

from surfaces._search_data_collection import SearchDataManager
from surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor import (
    GradientBoostingRegressorFunction,
)


class TestMLFunctionIntegration:
    """Integration tests for ML function search data functionality."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test databases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_gradient_boosting_regressor_basic_integration(self, temp_data_dir):
        """Test basic integration with GradientBoostingRegressorFunction."""
        # Mock the SearchDataManager to use our temp directory in the ML function
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            mock_manager_class.return_value = SearchDataManager(data_dir=temp_data_dir)

            func = GradientBoostingRegressorFunction()

            # Test normal evaluation
            test_params = {
                "n_estimators": 10,
                "max_depth": 3,
                "cv": 2,
                "dataset": func.dataset_default[0],
            }

            result_normal = func(test_params)
            assert isinstance(result_normal, float)

            # Test search data collection with small space
            small_search_space = {
                "n_estimators": [10, 20],
                "max_depth": [3, 5],
                "cv": [2],
                "dataset": func.dataset_default[:1],
            }

            collection_stats = func._collect_search_data(
                search_space=small_search_space, verbose=False
            )

            assert collection_stats["evaluations_collected"] == 4
            assert collection_stats["total_evaluations_stored"] == 4
            assert collection_stats["collection_time_seconds"] > 0

            # Test timing statistics
            timing = func._get_timing_statistics()
            assert timing["count"] == 4
            assert timing["average_time"] > 0

    def test_ml_function_with_evaluate_from_data(self, temp_data_dir):
        """Test ML function with evaluate_from_data=True."""
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            mock_manager_class.return_value = SearchDataManager(data_dir=temp_data_dir)

            # First collect some data
            func_collector = GradientBoostingRegressorFunction()
            small_search_space = {
                "n_estimators": [10],
                "max_depth": [3],
                "cv": [2],
                "dataset": func_collector.dataset_default[:1],
            }

            func_collector._collect_search_data(search_space=small_search_space, verbose=False)

            # Now create function that uses stored data
            func_from_data = GradientBoostingRegressorFunction(evaluate_from_data=True)

            test_params = {
                "n_estimators": 10,
                "max_depth": 3,
                "cv": 2,
                "dataset": func_collector.dataset_default[0],
            }

            # Should find stored evaluation
            start_time = time.time()
            result = func_from_data._objective_function_from_data(test_params)
            lookup_time = time.time() - start_time

            assert isinstance(result, float)
            assert lookup_time < 0.1  # Should be fast

    def test_ml_function_evaluate_from_data_missing(self, temp_data_dir):
        """Test ML function when stored data is missing."""
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            mock_manager_class.return_value = SearchDataManager(data_dir=temp_data_dir)

            func = GradientBoostingRegressorFunction(evaluate_from_data=True)

            test_params = {
                "n_estimators": 999,  # Parameter combination that won't exist
                "max_depth": 999,
                "cv": 2,
                "dataset": func.dataset_default[0],
            }

            with pytest.raises(ValueError) as exc_info:
                func._objective_function_from_data(test_params)

            assert "No stored evaluation found" in str(exc_info.value)

    def test_function_name_attribute_usage(self, temp_data_dir):
        """Test that functions use their _name_ attribute correctly."""
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            manager = SearchDataManager(data_dir=temp_data_dir)
            mock_manager_class.return_value = manager

            func = GradientBoostingRegressorFunction()

            # Collect some data
            small_search_space = {
                "n_estimators": [10],
                "max_depth": [3],
                "cv": [2],
                "dataset": func.dataset_default[:1],
            }

            func._collect_search_data(search_space=small_search_space, verbose=False)

            # Check that database was created with correct name
            expected_db_path = manager.get_db_path(func._name_)
            assert expected_db_path.endswith("gradient_boosting_regressor.db")

    def test_clear_search_data_functionality(self, temp_data_dir):
        """Test clearing search data functionality."""
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            mock_manager_class.return_value = SearchDataManager(data_dir=temp_data_dir)

            func = GradientBoostingRegressorFunction()

            # Collect some data
            small_search_space = {
                "n_estimators": [10],
                "max_depth": [3],
                "cv": [2],
                "dataset": func.dataset_default[:1],
            }

            func._collect_search_data(search_space=small_search_space, verbose=False)

            # Verify data exists by checking timing stats
            timing_before = func._get_timing_statistics()
            assert timing_before["count"] == 1

            # Clear data
            func._clear_search_data()

            # Verify data is cleared
            timing_after = func._get_timing_statistics()
            assert timing_after["count"] == 0

    def test_evaluation_type_consistency(self, temp_data_dir):
        """Test that evaluation results are consistent between normal and data lookup."""
        with patch(
            "surfaces.test_functions.machine_learning._base_machine_learning.SearchDataManager"
        ) as mock_manager_class:
            mock_manager_class.return_value = SearchDataManager(data_dir=temp_data_dir)

            func = GradientBoostingRegressorFunction()

            test_params = {
                "n_estimators": 10,
                "max_depth": 3,
                "cv": 2,
                "dataset": func.dataset_default[0],
            }

            # Get result from normal evaluation
            result_normal = func(test_params)

            # Collect data including this parameter combination
            small_search_space = {
                "n_estimators": [10],
                "max_depth": [3],
                "cv": [2],
                "dataset": func.dataset_default[:1],
            }

            func._collect_search_data(search_space=small_search_space, verbose=False)

            # Get result from data lookup
            func_from_data = GradientBoostingRegressorFunction(evaluate_from_data=True)
            result_lookup = func_from_data._objective_function_from_data(test_params)

            # Results should be very close (allowing for small differences due to dataset serialization)
            assert abs(result_normal - result_lookup) < 1e-3
