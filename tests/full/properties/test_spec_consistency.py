"""Property tests for function spec consistency (CEC functions).

These tests verify that CEC function specifications (_spec) are
properly defined.
"""

import inspect

import pytest

import surfaces.test_functions.benchmark.cec.cec2014 as cec2014
from tests.conftest import func_id, instantiate_function

CEC2014_FUNCTIONS = [
    v
    for k, v in vars(cec2014).items()
    if inspect.isclass(v) and not k.startswith("_") and k != "CEC2014Function"
]
CEC2014_UNIMODAL = CEC2014_FUNCTIONS[:3]


# =============================================================================
# Spec Existence and Structure
# =============================================================================


class TestSpecStructure:
    """Test that specs have correct structure."""

    @pytest.mark.cec
    @pytest.mark.parametrize("func_class", CEC2014_UNIMODAL, ids=func_id)
    def test_cec_has_func_id(self, func_class):
        """CEC functions have func_id in spec."""
        func = instantiate_function(func_class, n_dim=10)
        assert "func_id" in func.spec
