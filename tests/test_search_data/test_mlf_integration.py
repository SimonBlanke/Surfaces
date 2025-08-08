import shutil
import tempfile
import unittest
import numpy as np

from surfaces.data_collector_new import SearchDataManager
from surfaces.test_functions.machine_learning._base_machine_learning import (
    MachineLearningFunction,
)


class QuadraticMLFunc(MachineLearningFunction):
    _name_ = "QuadraticML"
    param_names = ["a", "b"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _actual_evaluate(self, params) -> float:
        # dummy expensive model
        if isinstance(params, dict):
            # Convert dict to array using param_names order
            x = np.array([params[name] for name in self.param_names])
        else:
            x = np.array(params)
        return float(np.sum(x**2))

    def create_objective_function(self):
        self.pure_objective_function = self._actual_evaluate


class MLFIntegrationTest(unittest.TestCase):

    def setUp(self):
        self.tmp_root = tempfile.mkdtemp()
        SearchDataManager._SUBDIR = self.tmp_root  # isolate FS side-effects

    def tearDown(self):
        shutil.rmtree(self.tmp_root)

    # ------------------------------------------------------------------
    def test_collect_then_lookup(self):
        space = {"a": [0.0, 1.0], "b": [0.0, 2.0]}
        f = QuadraticMLFunc(collect_search_data=True, search_space=space)

        # value inside the grid should be fetched from DB
        self.assertEqual(f.evaluate([1.0, 2.0]), 5.0)

        # unseen value falls back to computation
        self.assertAlmostEqual(f.evaluate([0.3, 0.4]), 0.3**2 + 0.4**2, places=6)


if __name__ == "__main__":
    unittest.main()
