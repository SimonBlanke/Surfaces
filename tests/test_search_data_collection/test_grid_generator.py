# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import pytest
from itertools import product
from surfaces.search_data_collection import GridGenerator


class TestGridGenerator:
    """Test suite for GridGenerator class."""
    
    def test_generate_grid_simple(self):
        """Test basic grid generation."""
        search_space = {
            'param1': [1, 2],
            'param2': ['a', 'b']
        }
        
        grid = GridGenerator.generate_grid(search_space)
        
        expected = [
            {'param1': 1, 'param2': 'a'},
            {'param1': 1, 'param2': 'b'},
            {'param1': 2, 'param2': 'a'},
            {'param1': 2, 'param2': 'b'}
        ]
        
        assert len(grid) == 4
        for expected_combo in expected:
            assert expected_combo in grid
    
    def test_generate_grid_single_param(self):
        """Test grid generation with single parameter."""
        search_space = {'param1': [1, 2, 3]}
        
        grid = GridGenerator.generate_grid(search_space)
        
        expected = [
            {'param1': 1},
            {'param1': 2},
            {'param1': 3}
        ]
        
        assert len(grid) == 3
        assert grid == expected
    
    def test_generate_grid_multiple_params(self):
        """Test grid generation with multiple parameters."""
        search_space = {
            'n_estimators': [10, 20],
            'max_depth': [3, 5],
            'cv': [2, 3, 4]
        }
        
        grid = GridGenerator.generate_grid(search_space)
        
        # Should have 2 * 2 * 3 = 12 combinations
        assert len(grid) == 12
        
        # Verify all combinations are present
        expected_combinations = list(product([10, 20], [3, 5], [2, 3, 4]))
        for n_est, max_d, cv in expected_combinations:
            expected_dict = {'n_estimators': n_est, 'max_depth': max_d, 'cv': cv}
            assert expected_dict in grid
    
    def test_generate_grid_empty(self):
        """Test grid generation with empty search space."""
        search_space = {}
        grid = GridGenerator.generate_grid(search_space)
        assert grid == []
    
    def test_generate_grid_iterator(self):
        """Test grid generation using iterator."""
        search_space = {
            'param1': [1, 2],
            'param2': ['a', 'b']
        }
        
        grid_list = list(GridGenerator.generate_grid_iterator(search_space))
        
        expected = [
            {'param1': 1, 'param2': 'a'},
            {'param1': 1, 'param2': 'b'},
            {'param1': 2, 'param2': 'a'},
            {'param1': 2, 'param2': 'b'}
        ]
        
        assert len(grid_list) == 4
        for expected_combo in expected:
            assert expected_combo in grid_list
    
    def test_generate_grid_iterator_empty(self):
        """Test grid iterator with empty search space."""
        search_space = {}
        grid_list = list(GridGenerator.generate_grid_iterator(search_space))
        assert grid_list == []
    
    def test_generate_grid_iterator_memory_efficiency(self):
        """Test that iterator doesn't load all combinations into memory at once."""
        search_space = {
            'param1': list(range(100)),
            'param2': list(range(50))
        }
        
        # Should be able to create iterator without memory issues
        iterator = GridGenerator.generate_grid_iterator(search_space)
        
        # Test first few items
        first_item = next(iterator)
        assert first_item == {'param1': 0, 'param2': 0}
        
        second_item = next(iterator)
        assert second_item == {'param1': 0, 'param2': 1}
    
    def test_count_combinations_simple(self):
        """Test counting combinations."""
        search_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b']
        }
        
        count = GridGenerator.count_combinations(search_space)
        assert count == 6  # 3 * 2
    
    def test_count_combinations_multiple_params(self):
        """Test counting with multiple parameters."""
        search_space = {
            'n_estimators': [10, 20, 30, 40, 50],  # 5 values
            'max_depth': [3, 5, 7],                # 3 values
            'cv': [2, 3, 4, 5]                     # 4 values
        }
        
        count = GridGenerator.count_combinations(search_space)
        assert count == 60  # 5 * 3 * 4
    
    def test_count_combinations_empty(self):
        """Test counting combinations with empty search space."""
        search_space = {}
        count = GridGenerator.count_combinations(search_space)
        assert count == 0
    
    def test_count_combinations_single_param(self):
        """Test counting combinations with single parameter."""
        search_space = {'param1': [1, 2, 3, 4, 5]}
        count = GridGenerator.count_combinations(search_space)
        assert count == 5
    
    def test_validate_search_space_valid(self):
        """Test validation of valid search space."""
        search_space = {
            'param1': [1, 2, 3],
            'param2': ['a', 'b', 'c']
        }
        
        errors = GridGenerator.validate_search_space(search_space)
        assert errors == []
    
    def test_validate_search_space_not_dict(self):
        """Test validation with non-dictionary input."""
        errors = GridGenerator.validate_search_space("not a dict")
        assert len(errors) == 1
        assert "must be a dictionary" in errors[0]
    
    def test_validate_search_space_empty(self):
        """Test validation of empty search space."""
        errors = GridGenerator.validate_search_space({})
        assert len(errors) == 1
        assert "cannot be empty" in errors[0]
    
    def test_validate_search_space_invalid_param_name(self):
        """Test validation with invalid parameter name."""
        search_space = {
            123: [1, 2, 3],  # Non-string key
            'valid_param': ['a', 'b']
        }
        
        errors = GridGenerator.validate_search_space(search_space)
        assert len(errors) == 1
        assert "must be string" in errors[0]
    
    def test_validate_search_space_invalid_param_values(self):
        """Test validation with invalid parameter values."""
        search_space = {
            'param1': "not a list",  # Should be a list
            'param2': [1, 2, 3]
        }
        
        errors = GridGenerator.validate_search_space(search_space)
        assert len(errors) == 1
        assert "must be a list" in errors[0]
    
    def test_validate_search_space_empty_param_values(self):
        """Test validation with empty parameter values."""
        search_space = {
            'param1': [],  # Empty list
            'param2': [1, 2, 3]
        }
        
        errors = GridGenerator.validate_search_space(search_space)
        assert len(errors) == 1
        assert "cannot be empty" in errors[0]
    
    def test_validate_search_space_multiple_errors(self):
        """Test validation with multiple errors."""
        search_space = {
            123: [],  # Non-string key and empty list
            'param2': "not a list"  # Not a list
        }
        
        errors = GridGenerator.validate_search_space(search_space)
        assert len(errors) == 3  # Non-string key, empty list, not a list
    
    def test_get_search_space_info(self):
        """Test getting search space information."""
        search_space = {
            'n_estimators': [10, 20, 30],
            'max_depth': [3, 5],
            'cv': [2, 3, 4, 5]
        }
        
        info = GridGenerator.get_search_space_info(search_space)
        
        assert info['parameter_count'] == 3
        assert set(info['parameters']) == {'n_estimators', 'max_depth', 'cv'}
        assert info['total_combinations'] == 24  # 3 * 2 * 4
        assert info['parameter_sizes']['n_estimators'] == 3
        assert info['parameter_sizes']['max_depth'] == 2
        assert info['parameter_sizes']['cv'] == 4
    
    def test_get_search_space_info_empty(self):
        """Test getting info for empty search space."""
        search_space = {}
        
        info = GridGenerator.get_search_space_info(search_space)
        
        assert info['parameter_count'] == 0
        assert info['parameters'] == []
        assert info['total_combinations'] == 0
        assert info['parameter_sizes'] == {}
    
    def test_get_search_space_info_single_param(self):
        """Test getting info for single parameter."""
        search_space = {'param1': [1, 2, 3, 4, 5]}
        
        info = GridGenerator.get_search_space_info(search_space)
        
        assert info['parameter_count'] == 1
        assert info['parameters'] == ['param1']
        assert info['total_combinations'] == 5
        assert info['parameter_sizes']['param1'] == 5
    
    def test_grid_generation_with_different_types(self):
        """Test grid generation with different parameter types."""
        search_space = {
            'int_param': [1, 2],
            'float_param': [1.0, 2.5],
            'str_param': ['a', 'b'],
            'bool_param': [True, False]
        }
        
        grid = GridGenerator.generate_grid(search_space)
        
        # Should have 2^4 = 16 combinations
        assert len(grid) == 16
        
        # Check a specific combination
        expected_combo = {
            'int_param': 1,
            'float_param': 1.0,
            'str_param': 'a',
            'bool_param': True
        }
        assert expected_combo in grid
    
    def test_grid_generation_preserves_order(self):
        """Test that grid generation preserves parameter order."""
        search_space = {
            'param_c': [3],
            'param_a': [1],
            'param_b': [2]
        }
        
        grid = GridGenerator.generate_grid(search_space)
        
        assert len(grid) == 1
        combination = grid[0]
        
        # All parameters should be present
        assert set(combination.keys()) == {'param_c', 'param_a', 'param_b'}
        assert combination['param_c'] == 3
        assert combination['param_a'] == 1
        assert combination['param_b'] == 2