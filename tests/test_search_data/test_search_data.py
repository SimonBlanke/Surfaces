# file: tests/test_search_data.py
import os
import shutil
import tempfile
import unittest

import numpy as np

from surfaces.data_collector_new import (
    SearchDataCollector,
    SearchDataLookup,
    SearchDataManager,
)


class _DummyMLFunction:
    """Stand-in for a real MachineLearningFunction."""

    _name_ = "DummySumSquares"

    def evaluate(self, x) -> float:
        # computationally cheap surrogate
        if isinstance(x, dict):
            # Convert dict values to array
            values = np.array(list(x.values()))
        else:
            values = np.array(x)
        return float(np.sum(values**2))


class SearchDataTestCase(unittest.TestCase):

    def setUp(self):
        # isolate every run in a temporary directory
        self.tmpdir = tempfile.mkdtemp()
        SearchDataManager._SUBDIR = self.tmpdir  # monkey-patch root folder

        self.func = _DummyMLFunction()
        self.space = {"a": [0.0, 1.0], "b": [0.0, 2.0]}

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    # ----------------------------------------------------------------------
    def test_collection_and_lookup(self):
        # 1. collect ----------------------------------------------------------------
        collector = SearchDataCollector(self.func.evaluate, self.space)
        data = collector.collect(verbose=False)
        
        # Verify data collection works
        self.assertEqual(len(data['scores']), 4)  # 2x2 grid
        self.assertTrue(all(isinstance(s, (int, float)) for s in data['scores']))
        
        # Test some known values: 
        # For grid points (0,0), (0,2), (1,0), (1,2) 
        # Expected results: 0, 4, 1, 5
        expected_scores = {(0.0, 0.0): 0.0, (0.0, 2.0): 4.0, (1.0, 0.0): 1.0, (1.0, 2.0): 5.0}
        
        for i, params in enumerate(data['parameters']):
            key = (params[0], params[1])
            if key in expected_scores:
                self.assertAlmostEqual(data['scores'][i], expected_scores[key], places=6)

    # ----------------------------------------------------------------------
    def test_manager_connection(self):
        path = SearchDataManager.get_db_path("SomeName")
        self.assertTrue(path.endswith(".sqlite3"))
        conn = SearchDataManager.connect("SomeName")
        conn.execute("CREATE TABLE IF NOT EXISTS t(x INT);")
        conn.close()


if __name__ == "__main__":
    unittest.main()
