# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import os
import sqlite3
import tempfile
import pytest
import time
from typing import Dict, Any

from surfaces._search_data_collection import SearchDataManager


class TestSearchDataManager:
    """Test suite for SearchDataManager class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test databases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def manager(self, temp_data_dir):
        """Create a SearchDataManager instance with temporary directory."""
        return SearchDataManager(data_dir=temp_data_dir)
    
    def test_manager_initialization(self, temp_data_dir):
        """Test SearchDataManager initialization and directory creation."""
        manager = SearchDataManager(data_dir=temp_data_dir)
        assert os.path.exists(temp_data_dir)
        assert manager.data_dir == temp_data_dir
    
    def test_default_data_dir_creation(self):
        """Test that default data directory is created correctly."""
        manager = SearchDataManager()
        expected_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "src", "surfaces", "search_data"
        )
        # Directory should be created
        assert os.path.exists(manager.data_dir)
    
    def test_get_db_path(self, manager, temp_data_dir):
        """Test database path generation."""
        function_name = "test_function"
        expected_path = os.path.join(temp_data_dir, f"{function_name}.db")
        actual_path = manager.get_db_path(function_name)
        assert actual_path == expected_path
    
    def test_create_table(self, manager):
        """Test table creation with various parameter configurations."""
        function_name = "test_create_table"
        parameter_names = ["param1", "param2", "param3"]
        
        # Create table
        manager.create_table(function_name, parameter_names)
        
        # Verify table exists and has correct structure
        db_path = manager.get_db_path(function_name)
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(search_data)")
            columns = cursor.fetchall()
            
            column_names = [col[1] for col in columns]
            expected_columns = ["id"] + parameter_names + ["score", "evaluation_time", "timestamp"]
            
            assert len(column_names) == len(expected_columns)
            for expected_col in expected_columns:
                assert expected_col in column_names
    
    def test_store_and_lookup_evaluation(self, manager):
        """Test storing and retrieving evaluations."""
        function_name = "test_store_lookup"
        parameter_names = ["n_estimators", "max_depth"]
        
        # Create table
        manager.create_table(function_name, parameter_names)
        
        # Store evaluation
        parameters = {"n_estimators": 100, "max_depth": 5}
        score = 0.95
        eval_time = 0.123
        
        manager.store_evaluation(function_name, parameters, score, eval_time)
        
        # Lookup evaluation
        result = manager.lookup_evaluation(function_name, parameters)
        assert result is not None
        retrieved_score, retrieved_time = result
        assert retrieved_score == score
        assert retrieved_time == eval_time
    
    def test_store_evaluation_with_function_objects(self, manager):
        """Test storing evaluations with function objects as parameters."""
        function_name = "test_function_objects"
        parameter_names = ["dataset", "n_estimators"]
        
        manager.create_table(function_name, parameter_names)
        
        # Use a function object as a parameter value
        def dummy_dataset():
            return "data"
        
        parameters = {"dataset": dummy_dataset, "n_estimators": 50}
        score = 0.88
        eval_time = 0.045
        
        manager.store_evaluation(function_name, parameters, score, eval_time)
        
        # Lookup should work with function name stored
        result = manager.lookup_evaluation(function_name, parameters)
        assert result is not None
        retrieved_score, retrieved_time = result
        assert retrieved_score == score
        assert retrieved_time == eval_time
    
    def test_store_batch(self, manager):
        """Test batch storage of evaluations."""
        function_name = "test_batch"
        parameter_names = ["param1", "param2"]
        
        manager.create_table(function_name, parameter_names)
        
        # Create batch of evaluations
        evaluations = [
            ({"param1": "value1", "param2": "value2"}, 0.1, 0.01),
            ({"param1": "value3", "param2": "value4"}, 0.2, 0.02),
            ({"param1": "value5", "param2": "value6"}, 0.3, 0.03),
        ]
        
        manager.store_batch(function_name, parameter_names, evaluations)
        
        # Verify all evaluations were stored
        all_evaluations = manager.get_all_evaluations(function_name)
        assert len(all_evaluations) == 3
        
        # Verify individual lookups work
        for params, expected_score, expected_time in evaluations:
            result = manager.lookup_evaluation(function_name, params)
            assert result is not None
            score, eval_time = result
            assert score == expected_score
            assert eval_time == expected_time
    
    def test_lookup_nonexistent_evaluation(self, manager):
        """Test lookup of non-existent evaluation returns None."""
        function_name = "test_nonexistent"
        parameter_names = ["param1"]
        
        manager.create_table(function_name, parameter_names)
        
        # Try to lookup non-existent evaluation
        result = manager.lookup_evaluation(function_name, {"param1": "nonexistent"})
        assert result is None
    
    def test_lookup_nonexistent_database(self, manager):
        """Test lookup from non-existent database returns None."""
        result = manager.lookup_evaluation("nonexistent_function", {"param": "value"})
        assert result is None
    
    def test_get_all_evaluations(self, manager):
        """Test retrieving all evaluations for a function."""
        function_name = "test_get_all"
        parameter_names = ["param1"]
        
        manager.create_table(function_name, parameter_names)
        
        # Store multiple evaluations
        evaluations_data = [
            ({"param1": "value1"}, 0.1, 0.01),
            ({"param1": "value2"}, 0.2, 0.02),
            ({"param1": "value3"}, 0.3, 0.03),
        ]
        
        for params, score, eval_time in evaluations_data:
            manager.store_evaluation(function_name, params, score, eval_time)
        
        # Get all evaluations
        all_evaluations = manager.get_all_evaluations(function_name)
        assert len(all_evaluations) == 3
        
        # Verify structure of returned data
        for evaluation in all_evaluations:
            assert "id" in evaluation
            assert "param1" in evaluation
            assert "score" in evaluation
            assert "evaluation_time" in evaluation
            assert "timestamp" in evaluation
    
    def test_get_all_evaluations_empty_database(self, manager):
        """Test getting all evaluations from empty database."""
        result = manager.get_all_evaluations("nonexistent_function")
        assert result == []
    
    def test_clear_data(self, manager):
        """Test clearing all data for a function."""
        function_name = "test_clear"
        parameter_names = ["param1"]
        
        manager.create_table(function_name, parameter_names)
        
        # Store some data
        manager.store_evaluation(function_name, {"param1": "value1"}, 0.5, 0.1)
        manager.store_evaluation(function_name, {"param1": "value2"}, 0.6, 0.2)
        
        # Verify data exists
        all_data = manager.get_all_evaluations(function_name)
        assert len(all_data) == 2
        
        # Clear data
        manager.clear_data(function_name)
        
        # Verify data is cleared
        all_data = manager.get_all_evaluations(function_name)
        assert len(all_data) == 0
    
    def test_get_database_info(self, manager):
        """Test getting database information."""
        function_name = "test_info"
        parameter_names = ["param1", "param2"]
        
        # Test non-existent database
        info = manager.get_database_info("nonexistent")
        assert info["exists"] is False
        
        # Create database and add data
        manager.create_table(function_name, parameter_names)
        manager.store_evaluation(function_name, {"param1": "v1", "param2": "v2"}, 0.5, 0.1)
        
        # Get database info
        info = manager.get_database_info(function_name)
        assert info["exists"] is True
        assert "path" in info
        assert "size_bytes" in info
        assert info["record_count"] == 1
        assert set(info["columns"]) >= {"id", "param1", "param2", "score", "evaluation_time", "timestamp"}
    
    def test_concurrent_access(self, manager):
        """Test basic concurrent access to database."""
        function_name = "test_concurrent"
        parameter_names = ["param1"]
        
        manager.create_table(function_name, parameter_names)
        
        # Store data
        manager.store_evaluation(function_name, {"param1": "value1"}, 0.5, 0.1)
        
        # Create another manager instance (simulating concurrent access)
        manager2 = SearchDataManager(data_dir=manager.data_dir)
        
        # Both should be able to read the data
        result1 = manager.lookup_evaluation(function_name, {"param1": "value1"})
        result2 = manager2.lookup_evaluation(function_name, {"param1": "value1"})
        
        assert result1 == result2
        assert result1 is not None
    
    def test_parameter_type_handling(self, manager):
        """Test handling of different parameter types."""
        function_name = "test_types"
        parameter_names = ["int_param", "float_param", "str_param", "bool_param"]
        
        manager.create_table(function_name, parameter_names)
        
        # Test different parameter types
        parameters = {
            "int_param": 42,
            "float_param": 3.14,
            "str_param": "test_string",
            "bool_param": True
        }
        
        manager.store_evaluation(function_name, parameters, 0.75, 0.05)
        
        # Lookup should work (all values stored as strings)
        result = manager.lookup_evaluation(function_name, parameters)
        assert result is not None
        score, eval_time = result
        assert score == 0.75
        assert eval_time == 0.05