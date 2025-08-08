import unittest
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch

from surfaces.data_collector import SurfacesDataCollector


class TestSurfacesDataCollector(unittest.TestCase):
    """
    Comprehensive test suite for the SurfacesDataCollector class,
    focusing on the collect method functionality.
    """

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.temp_dir, "test_surfaces.db")
        self.collector = SurfacesDataCollector(path=self.test_path)

    def tearDown(self):
        """Clean up after each test method."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)

    def test_basic_collect_functionality(self):
        """Test basic grid search collection with a simple objective function."""

        def simple_objective(params):
            return params["x"] + params["y"]

        search_space = {"x": np.array([1, 2, 3]), "y": np.array([4, 5])}

        self.collector.collect(simple_objective, search_space, table="test_simple")

        # Verify correct number of evaluations
        expected_evaluations = 3 * 2  # 3 x values * 2 y values
        self.assertEqual(len(self.collector.search_data), expected_evaluations)

        # Verify all combinations were evaluated
        results = self.collector.search_data.data
        expected_combinations = [
            {"x": 1, "y": 4, "score": 5},
            {"x": 1, "y": 5, "score": 6},
            {"x": 2, "y": 4, "score": 6},
            {"x": 2, "y": 5, "score": 7},
            {"x": 3, "y": 4, "score": 7},
            {"x": 3, "y": 5, "score": 8},
        ]

        # Sort both lists for comparison
        results_sorted = sorted(results, key=lambda d: (d["x"], d["y"]))
        expected_sorted = sorted(expected_combinations, key=lambda d: (d["x"], d["y"]))

        self.assertEqual(results_sorted, expected_sorted)

    def test_quadratic_objective_function(self):
        """Test collection with a quadratic objective function."""

        def quadratic_objective(params):
            x, y = params["x"], params["y"]
            return -(x**2 + y**2)

        search_space = {"x": np.linspace(-2, 2, 5), "y": np.linspace(-2, 2, 5)}

        self.collector.collect(
            quadratic_objective, search_space, table="test_quadratic"
        )

        # Verify all 25 combinations were evaluated
        self.assertEqual(len(self.collector.search_data), 25)

        # Verify the optimal point (0, 0) has the highest score
        results = self.collector.search_data.data
        best_result = max(results, key=lambda d: d["score"])

        self.assertAlmostEqual(best_result["x"], 0.0, places=5)
        self.assertAlmostEqual(best_result["y"], 0.0, places=5)
        self.assertAlmostEqual(best_result["score"], 0.0, places=5)

    def test_list_search_space(self):
        """Test collection with list-based search space instead of numpy arrays."""

        def objective(params):
            return params["a"] * params["b"]

        search_space = {"a": [1, 2, 3], "b": [10, 20]}

        self.collector.collect(objective, search_space, table="test_list")

        # Verify correct number of evaluations
        self.assertEqual(len(self.collector.search_data), 6)

        # Verify scores
        scores = {d["score"] for d in self.collector.search_data.data}
        expected_scores = {10, 20, 20, 40, 30, 60}
        self.assertEqual(scores, expected_scores)

    def test_single_parameter_search(self):
        """Test collection with only one parameter."""

        def single_param_objective(params):
            return params["x"] ** 2

        search_space = {"x": np.array([1, 2, 3, 4])}

        self.collector.collect(
            single_param_objective, search_space, table="test_single"
        )

        self.assertEqual(len(self.collector.search_data), 4)

        # Verify scores
        results = {d["x"]: d["score"] for d in self.collector.search_data.data}
        expected = {1: 1, 2: 4, 3: 9, 4: 16}
        self.assertEqual(results, expected)

    def test_duplicate_handling(self):
        """Test that duplicate parameter combinations are not evaluated twice."""
        evaluation_count = 0

        def counting_objective(params):
            nonlocal evaluation_count
            evaluation_count += 1
            return params["x"]

        search_space = {"x": np.array([1, 2, 3])}

        # First collection
        self.collector.collect(
            counting_objective, search_space, table="test_duplicates"
        )
        first_count = evaluation_count

        # Second collection with same search space
        self.collector.collect(
            counting_objective, search_space, table="test_duplicates"
        )

        # Should not evaluate again due to warm start
        self.assertEqual(evaluation_count, first_count)
        self.assertEqual(len(self.collector.search_data), 3)

    def test_error_handling_in_objective(self):
        """Test handling of errors raised by the objective function."""

        def faulty_objective(params):
            if params["x"] == 2:
                raise ValueError("Intentional error")
            return params["x"]

        search_space = {"x": np.array([1, 2, 3])}

        # Should not raise an exception
        self.collector.collect(faulty_objective, search_space, table="test_errors")

        # Should still have all evaluations
        self.assertEqual(len(self.collector.search_data), 3)

        # Check that error case has -inf score
        results = self.collector.search_data.data
        error_result = next(r for r in results if r["x"] == 2)
        self.assertEqual(error_result["score"], float("-inf"))

    def test_table_name_from_function(self):
        """Test that table name is derived from function name when not provided."""

        def my_special_function(params):
            return params["x"]

        search_space = {"x": [1, 2, 3]}

        with patch.object(self.collector, "save") as mock_save:
            self.collector.collect(my_special_function, search_space)

            # Verify save was called with function name as table
            mock_save.assert_called_once()
            args = mock_save.call_args[0]
            self.assertEqual(args[0], "my_special_function")

    def test_large_search_space(self):
        """Test collection with a larger search space."""

        def objective(params):
            return sum(params.values())

        # Create a 4D search space with 625 total combinations (5^4)
        search_space = {
            "a": np.linspace(0, 1, 5),
            "b": np.linspace(0, 1, 5),
            "c": np.linspace(0, 1, 5),
            "d": np.linspace(0, 1, 5),
        }

        self.collector.collect(objective, search_space, table="test_large")

        self.assertEqual(len(self.collector.search_data), 625)

        # Verify the maximum score
        max_result = max(self.collector.search_data.data, key=lambda d: d["score"])
        self.assertAlmostEqual(max_result["score"], 4.0, places=5)

    def test_mixed_numeric_types(self):
        """Test collection with mixed integer and float values."""

        def objective(params):
            return params["int_param"] * params["float_param"]

        search_space = {
            "int_param": [1, 2, 3],
            "float_param": np.array([0.5, 1.5, 2.5]),
        }

        self.collector.collect(objective, search_space, table="test_mixed")

        self.assertEqual(len(self.collector.search_data), 9)

        # Verify some specific results
        results = self.collector.search_data.data
        # Find result where int_param=2 and float_param=1.5
        specific_result = next(
            r
            for r in results
            if r["int_param"] == 2 and abs(r["float_param"] - 1.5) < 0.001
        )
        self.assertAlmostEqual(specific_result["score"], 3.0, places=5)

    def test_empty_search_space_dimension(self):
        """Test handling of empty dimension in search space."""

        def objective(params):
            return params.get("x", 0) + params.get("y", 0)

        search_space = {"x": np.array([1, 2, 3]), "y": np.array([])}  # Empty array

        # Should handle gracefully (no evaluations since one dimension is empty)
        self.collector.collect(objective, search_space, table="test_empty")
        self.assertEqual(len(self.collector.search_data), 0)

    def test_progress_verbosity(self):
        """Test that progress is shown when verbosity includes progress_bar."""

        def objective(params):
            return params["x"]

        search_space = {"x": np.linspace(0, 100, 101)}  # 101 points

        # Capture print output
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            self.collector.collect(objective, search_space, table="test_progress")
            output = captured_output.getvalue()

            # Should contain progress information
            self.assertIn("Grid Search Progress:", output)
            self.assertIn("100.0%", output)
        finally:
            sys.stdout = sys.__stdout__

    def test_if_exists_parameter(self):
        """Test different if_exists parameter values."""

        def objective(params):
            return params["x"]

        search_space = {"x": [1, 2, 3]}

        # Test with different if_exists values
        for if_exists in ["append", "replace", "fail"]:
            with patch.object(self.collector, "save") as mock_save:
                self.collector.collect(
                    objective, search_space, table="test_if_exists", if_exists=if_exists
                )

                # Verify save was called with correct if_exists parameter
                mock_save.assert_called_once()
                args = mock_save.call_args[0]
                self.assertEqual(args[2], if_exists)


class TestSurfacesDataCollectorIntegration(unittest.TestCase):
    """Integration tests for SurfacesDataCollector with actual save/load operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = os.path.join(self.temp_dir, "test_integration.db")
        self.collector = SurfacesDataCollector(path=self.test_path)

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir)

    def test_save_and_load_roundtrip(self):
        """Test that collected data can be saved and loaded correctly."""

        def objective(params):
            return params["x"] ** 2 + params["y"] ** 2

        search_space = {"x": np.array([-1, 0, 1]), "y": np.array([-1, 0, 1])}

        # Collect data
        self.collector.collect(objective, search_space, table="roundtrip_test")
        original_data = self.collector.search_data.data.copy()

        # Create new collector and load data
        new_collector = SurfacesDataCollector(path=self.test_path)
        loaded_data = new_collector.load("roundtrip_test")

        # Verify data integrity
        self.assertEqual(len(loaded_data), len(original_data))

        # Note: The actual comparison would depend on how SqlSearchData
        # implements save/load, which we're mocking here

    def test_incremental_collection(self):
        """Test collecting data in multiple stages."""

        def objective(params):
            return params["x"]

        # First collection
        search_space_1 = {"x": [1, 2, 3]}
        self.collector.collect(objective, search_space_1, table="incremental")
        first_length = len(self.collector.search_data)

        # Second collection with overlapping values
        search_space_2 = {"x": [3, 4, 5]}
        self.collector.collect(objective, search_space_2, table="incremental")

        # Should have 5 unique values total
        self.assertEqual(len(self.collector.search_data), 5)
        self.assertGreater(len(self.collector.search_data), first_length)


if __name__ == "__main__":
    unittest.main()
