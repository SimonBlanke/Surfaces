# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import tempfile
import pytest
import os
from unittest.mock import Mock, patch

from surfaces.search_data_collection import SearchDataManager, SearchDataCollector
from surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor import GradientBoostingRegressorFunction


class MockMLFunction:
    """Mock ML function for testing."""
    
    def __init__(self, name="mock_function"):
        self._name_ = name
        self.evaluation_count = 0
        self.evaluation_times = []
    
    def search_space(self):
        return {
            'param1': [1, 2],
            'param2': ['a', 'b']
        }
    
    def pure_objective_function(self, params):
        """Mock evaluation that tracks calls."""
        self.evaluation_count += 1
        # Return different scores based on parameters
        if params['param1'] == 1 and params['param2'] == 'a':
            return 0.1
        elif params['param1'] == 1 and params['param2'] == 'b':
            return 0.2
        elif params['param1'] == 2 and params['param2'] == 'a':
            return 0.3
        else:  # params['param1'] == 2 and params['param2'] == 'b'
            return 0.4


class TestSearchDataCollector:
    """Test suite for SearchDataCollector class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test databases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def data_manager(self, temp_data_dir):
        """Create a SearchDataManager with temporary directory."""
        return SearchDataManager(data_dir=temp_data_dir)
    
    @pytest.fixture
    def collector(self, data_manager):
        """Create a SearchDataCollector instance."""
        return SearchDataCollector(data_manager=data_manager)
    
    @pytest.fixture
    def mock_function(self):
        """Create a mock ML function."""
        return MockMLFunction()
    
    def test_collector_initialization(self, temp_data_dir):
        """Test SearchDataCollector initialization."""
        # Test with provided data manager
        manager = SearchDataManager(data_dir=temp_data_dir)
        collector = SearchDataCollector(data_manager=manager)
        assert collector.data_manager == manager
        
        # Test with default data manager
        collector_default = SearchDataCollector()
        assert collector_default.data_manager is not None
    
    def test_collect_search_data_complete(self, collector, mock_function):
        """Test complete search data collection."""
        stats = collector.collect_search_data(mock_function, verbose=False)
        
        # Verify statistics
        assert stats['function_name'] == 'mock_function'
        assert stats['total_combinations'] == 4
        assert stats['evaluations_collected'] == 4
        assert stats['total_evaluations_stored'] == 4
        assert stats['collection_time_seconds'] > 0
        
        # Verify function was called correct number of times
        assert mock_function.evaluation_count == 4
        
        # Verify all combinations were stored
        stored_evaluations = collector.data_manager.get_all_evaluations('mock_function')
        assert len(stored_evaluations) == 4
    
    def test_collect_search_data_custom_search_space(self, collector, mock_function):
        """Test data collection with custom search space."""
        custom_search_space = {
            'param1': [1],
            'param2': ['a', 'b']
        }
        
        stats = collector.collect_search_data(
            mock_function, 
            search_space=custom_search_space,
            verbose=False
        )
        
        # Should only have 2 combinations (1 * 2)
        assert stats['total_combinations'] == 2
        assert stats['evaluations_collected'] == 2
        assert mock_function.evaluation_count == 2
    
    def test_collect_search_data_with_existing_data(self, collector, mock_function):
        """Test collection when some data already exists."""
        # First collection
        stats1 = collector.collect_search_data(mock_function, verbose=False)
        assert stats1['evaluations_collected'] == 4
        
        # Reset function call count
        mock_function.evaluation_count = 0
        
        # Second collection should not collect any new data
        stats2 = collector.collect_search_data(mock_function, verbose=False)
        assert stats2['evaluations_collected'] == 0
        assert stats2['total_evaluations_stored'] == 4
        assert mock_function.evaluation_count == 0  # No new evaluations
    
    def test_collect_search_data_batch_processing(self, collector, mock_function):
        """Test batch processing functionality."""
        # Use small batch size
        stats = collector.collect_search_data(
            mock_function, 
            batch_size=2,
            verbose=False
        )
        
        # Should still collect all data
        assert stats['evaluations_collected'] == 4
        assert stats['total_evaluations_stored'] == 4
    
    def test_collect_search_data_invalid_search_space(self, collector, mock_function):
        """Test collection with invalid search space."""
        invalid_search_space = {
            'param1': [],  # Empty parameter values
        }
        
        with pytest.raises(ValueError) as exc_info:
            collector.collect_search_data(
                mock_function,
                search_space=invalid_search_space,
                verbose=False
            )
        
        assert "Invalid search space" in str(exc_info.value)
    
    def test_evaluate_with_lookup_new_evaluation(self, collector, mock_function):
        """Test evaluation with lookup for new parameters."""
        parameters = {'param1': 1, 'param2': 'a'}
        
        # First evaluation should compute and store
        score, eval_time = collector.evaluate_with_lookup(mock_function, parameters)
        
        assert score == 0.1  # Expected score from mock function
        assert eval_time >= 0  # Allow for very fast evaluations
        assert mock_function.evaluation_count == 1
        
        # Verify it was stored
        stored_result = collector.data_manager.lookup_evaluation('mock_function', parameters)
        assert stored_result == (score, eval_time)
    
    def test_evaluate_with_lookup_existing_evaluation(self, collector, mock_function):
        """Test evaluation with lookup for existing parameters."""
        parameters = {'param1': 1, 'param2': 'a'}
        
        # First evaluation
        score1, eval_time1 = collector.evaluate_with_lookup(mock_function, parameters)
        original_eval_count = mock_function.evaluation_count
        
        # Second evaluation should use stored data
        score2, eval_time2 = collector.evaluate_with_lookup(mock_function, parameters)
        
        assert score2 == score1
        assert eval_time2 == eval_time1
        assert mock_function.evaluation_count == original_eval_count  # No new evaluation
    
    def test_get_collection_status(self, collector, mock_function):
        """Test getting collection status."""
        # Before collection
        status_before = collector.get_collection_status(mock_function)
        assert status_before['stored_evaluations'] == 0
        assert status_before['completion_percentage'] == 0.0
        
        # After partial collection
        custom_search_space = {'param1': [1], 'param2': ['a', 'b']}
        collector.collect_search_data(mock_function, search_space=custom_search_space, verbose=False)
        
        # Get status using original search space (4 combinations)
        status_after = collector.get_collection_status(mock_function)
        assert status_after['stored_evaluations'] == 2
        assert status_after['total_combinations'] == 4
        assert status_after['completion_percentage'] == 50.0
    
    def test_clear_function_data(self, collector, mock_function):
        """Test clearing function data."""
        # Collect some data
        collector.collect_search_data(mock_function, verbose=False)
        
        # Verify data exists
        status_before = collector.get_collection_status(mock_function)
        assert status_before['stored_evaluations'] > 0
        
        # Clear data
        collector.clear_function_data(mock_function)
        
        # Verify data is cleared
        status_after = collector.get_collection_status(mock_function)
        assert status_after['stored_evaluations'] == 0
    
    def test_get_timing_statistics_empty(self, collector, mock_function):
        """Test timing statistics with no data."""
        stats = collector.get_timing_statistics(mock_function)
        assert stats['count'] == 0
    
    def test_get_timing_statistics_with_data(self, collector, mock_function):
        """Test timing statistics with collected data."""
        # Collect data
        collector.collect_search_data(mock_function, verbose=False)
        
        # Get timing statistics
        stats = collector.get_timing_statistics(mock_function)
        
        assert stats['count'] == 4
        assert stats['total_time'] > 0
        assert stats['average_time'] > 0
        assert stats['min_time'] > 0
        assert stats['max_time'] >= stats['min_time']
        assert abs(stats['total_time'] - stats['average_time'] * stats['count']) < 1e-10
    
    def test_collection_with_function_evaluation_error(self, collector):
        """Test collection when function evaluation raises an error."""
        class ErrorFunction(MockMLFunction):
            def pure_objective_function(self, params):
                if params['param1'] == 2 and params['param2'] == 'b':
                    raise ValueError("Evaluation error")
                return super().pure_objective_function(params)
        
        error_function = ErrorFunction()
        
        # Collection should continue despite errors
        stats = collector.collect_search_data(error_function, verbose=False)
        
        # Should collect 3 out of 4 combinations (one failed)
        assert stats['evaluations_collected'] == 3
        assert stats['total_evaluations_stored'] == 3
    
    @patch('surfaces.search_data_collection.data_collector.time.time')
    def test_timing_measurement_accuracy(self, mock_time, collector, mock_function):
        """Test that timing measurements are accurate."""
        # Mock time.time() to return predictable values
        # Need more values: collection start, eval1 start, eval1 end, eval2 start, eval2 end, etc, batch store, collection end
        time_sequence = [0.0,  # collection start
                        1.0, 1.1,  # eval 1 (param1=1, param2=a)
                        2.0, 2.2,  # eval 2 (param1=1, param2=b)  
                        3.0, 3.2,  # eval 3 (param1=2, param2=a)
                        4.0, 4.3,  # eval 4 (param1=2, param2=b)
                        4.4,  # batch store timestamp
                        5.0]  # collection end
        mock_time.side_effect = time_sequence
        
        collector.collect_search_data(mock_function, verbose=False)
        
        # Get timing statistics
        stats = collector.get_timing_statistics(mock_function)
        
        # Expected times: 0.1, 0.2, 0.2, 0.3
        assert stats['count'] == 4
        assert abs(stats['min_time'] - 0.1) < 1e-10
        assert abs(stats['max_time'] - 0.3) < 1e-10
        assert abs(stats['total_time'] - 0.8) < 1e-10
        assert abs(stats['average_time'] - 0.2) < 1e-10


class TestIntegrationWithRealMLFunction:
    """Integration tests with real ML functions."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test databases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def collector(self, temp_data_dir):
        """Create a SearchDataCollector with temporary directory."""
        manager = SearchDataManager(data_dir=temp_data_dir)
        return SearchDataCollector(data_manager=manager)
    
    def test_real_ml_function_collection(self, collector):
        """Test collection with real ML function."""
        func = GradientBoostingRegressorFunction()
        
        # Use very small search space to keep test fast
        small_search_space = {
            'n_estimators': [10],
            'max_depth': [3],
            'cv': [2],
            'dataset': func.dataset_default[:1]
        }
        
        stats = collector.collect_search_data(
            func,
            search_space=small_search_space,
            verbose=False
        )
        
        assert stats['total_combinations'] == 1
        assert stats['evaluations_collected'] == 1
        assert stats['total_evaluations_stored'] == 1
        assert stats['collection_time_seconds'] > 0
    
    def test_real_ml_function_timing_statistics(self, collector):
        """Test timing statistics with real ML function."""
        func = GradientBoostingRegressorFunction()
        
        # Use small search space
        small_search_space = {
            'n_estimators': [10, 20],
            'max_depth': [3],
            'cv': [2],
            'dataset': func.dataset_default[:1]
        }
        
        collector.collect_search_data(
            func,
            search_space=small_search_space,
            verbose=False
        )
        
        timing_stats = collector.get_timing_statistics(func)
        
        assert timing_stats['count'] == 2
        assert timing_stats['average_time'] > 0
        assert timing_stats['min_time'] > 0
        assert timing_stats['max_time'] >= timing_stats['min_time']
    
    def test_real_ml_function_fast_lookup(self, collector):
        """Test fast lookup functionality with real ML function."""
        func = GradientBoostingRegressorFunction()
        
        # Small search space
        small_search_space = {
            'n_estimators': [10],
            'max_depth': [3],
            'cv': [2],
            'dataset': func.dataset_default[:1]
        }
        
        # Collect data
        collector.collect_search_data(
            func,
            search_space=small_search_space,
            verbose=False
        )
        
        # Test lookup
        test_params = {
            'n_estimators': 10,
            'max_depth': 3,
            'cv': 2,
            'dataset': func.dataset_default[0]
        }
        
        score, eval_time = collector.evaluate_with_lookup(func, test_params)
        
        assert isinstance(score, float)
        assert eval_time > 0