"""Property tests for search space definitions (CEC and ML functions).

These tests verify that CEC and ML functions have valid search spaces
with correct structure and bounds.
"""

import inspect

import pytest

import surfaces.test_functions.benchmark.cec.cec2014 as cec2014
from surfaces.test_functions.machine_learning import machine_learning_functions
from tests.conftest import func_id, instantiate_function

CEC2014_FUNCTIONS = [
    v
    for k, v in vars(cec2014).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2014Function"
]


class TestSearchSpaceStructure:
    """Test that search spaces have correct structure."""

    @pytest.mark.cec
    @pytest.mark.parametrize("func_class", CEC2014_FUNCTIONS[:5], ids=func_id)
    def test_cec_search_space_is_dict(self, func_class):
        """CEC functions have dict search space."""
        func = instantiate_function(func_class, n_dim=10)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) == 10

    @pytest.mark.ml
    @pytest.mark.parametrize("func_class", machine_learning_functions[:5], ids=func_id)
    def test_ml_search_space_is_dict(self, func_class):
        """ML functions have dict search space."""
        func = instantiate_function(func_class)
        assert isinstance(func.search_space, dict)
        assert len(func.search_space) > 0
