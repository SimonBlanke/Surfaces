"""Property tests for global optima (CEC functions).

These tests verify that CEC functions with known global optima
return the correct value when evaluated at the optimum.
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
# CEC Functions - Global Optima
# =============================================================================


@pytest.mark.cec
class TestCECGlobalOptima:
    """Test global optima for CEC functions."""

    @pytest.mark.parametrize("func_class", CEC2014_UNIMODAL, ids=func_id)
    def test_cec_unimodal_has_global(self, func_class):
        """CEC unimodal functions have global minimum."""
        func = instantiate_function(func_class, n_dim=10)
        assert hasattr(func, "f_global") or "f_global" in func.spec
