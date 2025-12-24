# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Centralized pytest configuration and fixtures for Surfaces test suite.

This module provides:
- Categorized function lists for parametrized testing
- Shared fixtures for common test patterns
- Helper functions for test utilities
"""

import pytest
import numpy as np

# =============================================================================
# Function Discovery and Categorization
# =============================================================================

# Algebraic functions (always available)
from surfaces.test_functions.algebraic import (
    algebraic_functions,
    algebraic_functions_1d,
    algebraic_functions_2d,
    algebraic_functions_nd,
)

# Engineering functions (always available)
from surfaces.test_functions.engineering import engineering_functions

# BBOB functions (always available)
from surfaces.test_functions.bbob import BBOB_FUNCTIONS

# CEC functions (require data packages)
try:
    from surfaces.test_functions.cec.cec2013 import (
        Sphere as CEC2013_Sphere,
        RotatedHighConditionedElliptic as CEC2013_RotatedHighConditionedElliptic,
        RotatedBentCigar as CEC2013_RotatedBentCigar,
        RotatedDiscus as CEC2013_RotatedDiscus,
        DifferentPowers as CEC2013_DifferentPowers,
        RotatedRosenbrock as CEC2013_RotatedRosenbrock,
        RotatedSchafferF7 as CEC2013_RotatedSchafferF7,
        RotatedAckley as CEC2013_RotatedAckley,
        RotatedWeierstrass as CEC2013_RotatedWeierstrass,
        RotatedGriewank as CEC2013_RotatedGriewank,
        Rastrigin as CEC2013_Rastrigin,
        RotatedRastrigin as CEC2013_RotatedRastrigin,
        StepRastrigin as CEC2013_StepRastrigin,
        Schwefel as CEC2013_Schwefel,
        RotatedSchwefel as CEC2013_RotatedSchwefel,
        RotatedKatsuura as CEC2013_RotatedKatsuura,
        LunacekBiRastrigin as CEC2013_LunacekBiRastrigin,
        RotatedLunacekBiRastrigin as CEC2013_RotatedLunacekBiRastrigin,
        RotatedExpandedGriewankRosenbrock as CEC2013_RotatedExpandedGriewankRosenbrock,
        RotatedExpandedScafferF6 as CEC2013_RotatedExpandedScafferF6,
        CompositionFunction1 as CEC2013_CompositionFunction1,
        CompositionFunction2 as CEC2013_CompositionFunction2,
        CompositionFunction3 as CEC2013_CompositionFunction3,
        CompositionFunction4 as CEC2013_CompositionFunction4,
        CompositionFunction5 as CEC2013_CompositionFunction5,
        CompositionFunction6 as CEC2013_CompositionFunction6,
        CompositionFunction7 as CEC2013_CompositionFunction7,
        CompositionFunction8 as CEC2013_CompositionFunction8,
    )

    CEC2013_FUNCTIONS = [
        # Unimodal (F1-F5)
        CEC2013_Sphere,
        CEC2013_RotatedHighConditionedElliptic,
        CEC2013_RotatedBentCigar,
        CEC2013_RotatedDiscus,
        CEC2013_DifferentPowers,
        # Multimodal (F6-F20)
        CEC2013_RotatedRosenbrock,
        CEC2013_RotatedSchafferF7,
        CEC2013_RotatedAckley,
        CEC2013_RotatedWeierstrass,
        CEC2013_RotatedGriewank,
        CEC2013_Rastrigin,
        CEC2013_RotatedRastrigin,
        CEC2013_StepRastrigin,
        CEC2013_Schwefel,
        CEC2013_RotatedSchwefel,
        CEC2013_RotatedKatsuura,
        CEC2013_LunacekBiRastrigin,
        CEC2013_RotatedLunacekBiRastrigin,
        CEC2013_RotatedExpandedGriewankRosenbrock,
        CEC2013_RotatedExpandedScafferF6,
        # Composition (F21-F28)
        CEC2013_CompositionFunction1,
        CEC2013_CompositionFunction2,
        CEC2013_CompositionFunction3,
        CEC2013_CompositionFunction4,
        CEC2013_CompositionFunction5,
        CEC2013_CompositionFunction6,
        CEC2013_CompositionFunction7,
        CEC2013_CompositionFunction8,
    ]
    HAS_CEC2013 = True
except ImportError:
    CEC2013_FUNCTIONS = []
    HAS_CEC2013 = False

try:
    from surfaces.test_functions.cec.cec2014 import (
        RotatedHighConditionedElliptic,
        RotatedBentCigar,
        RotatedDiscus,
        ShiftedRotatedRosenbrock,
        ShiftedRotatedAckley,
        ShiftedRotatedWeierstrass,
        ShiftedRotatedGriewank,
        ShiftedRastrigin,
        ShiftedRotatedRastrigin,
        ShiftedSchwefel,
        ShiftedRotatedSchwefel,
        ShiftedRotatedKatsuura,
        ShiftedRotatedHappyCat,
        ShiftedRotatedHGBat,
        ShiftedRotatedExpandedGriewankRosenbrock,
        ShiftedRotatedExpandedScafferF6,
        HybridFunction1,
        HybridFunction2,
        HybridFunction3,
        HybridFunction4,
        HybridFunction5,
        HybridFunction6,
        CompositionFunction1,
        CompositionFunction2,
        CompositionFunction3,
        CompositionFunction4,
        CompositionFunction5,
        CompositionFunction6,
        CompositionFunction7,
        CompositionFunction8,
    )

    CEC2014_FUNCTIONS = [
        # Unimodal (F1-F3)
        RotatedHighConditionedElliptic,
        RotatedBentCigar,
        RotatedDiscus,
        # Multimodal (F4-F16)
        ShiftedRotatedRosenbrock,
        ShiftedRotatedAckley,
        ShiftedRotatedWeierstrass,
        ShiftedRotatedGriewank,
        ShiftedRastrigin,
        ShiftedRotatedRastrigin,
        ShiftedSchwefel,
        ShiftedRotatedSchwefel,
        ShiftedRotatedKatsuura,
        ShiftedRotatedHappyCat,
        ShiftedRotatedHGBat,
        ShiftedRotatedExpandedGriewankRosenbrock,
        ShiftedRotatedExpandedScafferF6,
        # Hybrid (F17-F22)
        HybridFunction1,
        HybridFunction2,
        HybridFunction3,
        HybridFunction4,
        HybridFunction5,
        HybridFunction6,
        # Composition (F23-F30)
        CompositionFunction1,
        CompositionFunction2,
        CompositionFunction3,
        CompositionFunction4,
        CompositionFunction5,
        CompositionFunction6,
        CompositionFunction7,
        CompositionFunction8,
    ]
    CEC2014_UNIMODAL = CEC2014_FUNCTIONS[:3]
    CEC2014_MULTIMODAL = CEC2014_FUNCTIONS[3:16]
    CEC2014_HYBRID = CEC2014_FUNCTIONS[16:22]
    CEC2014_COMPOSITION = CEC2014_FUNCTIONS[22:]
    HAS_CEC2014 = True
except ImportError:
    CEC2014_FUNCTIONS = []
    CEC2014_UNIMODAL = []
    CEC2014_MULTIMODAL = []
    CEC2014_HYBRID = []
    CEC2014_COMPOSITION = []
    HAS_CEC2014 = False

try:
    from surfaces.test_functions.cec.cec2017 import (
        ShiftedRotatedBentCigar,
        ShiftedRotatedSumDiffPow,
        ShiftedRotatedZakharov,
        ShiftedRotatedRosenbrock as CEC2017_ShiftedRotatedRosenbrock,
        ShiftedRotatedRastrigin as CEC2017_ShiftedRotatedRastrigin,
        ShiftedRotatedSchafferF7 as CEC2017_ShiftedRotatedSchafferF7,
        ShiftedRotatedLunacekBiRastrigin,
        ShiftedRotatedNonContRastrigin,
        ShiftedRotatedLevy,
        ShiftedRotatedSchwefel as CEC2017_ShiftedRotatedSchwefel,
    )

    CEC2017_FUNCTIONS = [
        ShiftedRotatedBentCigar,
        ShiftedRotatedSumDiffPow,
        ShiftedRotatedZakharov,
        CEC2017_ShiftedRotatedRosenbrock,
        CEC2017_ShiftedRotatedRastrigin,
        CEC2017_ShiftedRotatedSchafferF7,
        ShiftedRotatedLunacekBiRastrigin,
        ShiftedRotatedNonContRastrigin,
        ShiftedRotatedLevy,
        CEC2017_ShiftedRotatedSchwefel,
    ]
    HAS_CEC2017 = True
except ImportError:
    CEC2017_FUNCTIONS = []
    HAS_CEC2017 = False

# Machine learning functions (require sklearn)
try:
    from surfaces.test_functions.machine_learning import machine_learning_functions

    HAS_ML = True
except ImportError:
    machine_learning_functions = []
    HAS_ML = False

# Visualization dependencies
try:
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False
    go = None
    plt = None

# Streamlit testing
try:
    from streamlit.testing.v1 import AppTest

    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    AppTest = None

# BBOB as list
BBOB_FUNCTION_LIST = list(BBOB_FUNCTIONS.values())

# All functions combined
ALL_ALGEBRAIC = algebraic_functions
ALL_ENGINEERING = engineering_functions
ALL_BBOB = BBOB_FUNCTION_LIST
ALL_CEC = CEC2013_FUNCTIONS + CEC2014_FUNCTIONS + CEC2017_FUNCTIONS
ALL_ML = machine_learning_functions if HAS_ML else []

# Functions with known global optima
FUNCTIONS_WITH_GLOBAL_OPTIMA = [
    f for f in (ALL_ALGEBRAIC + ALL_ENGINEERING + ALL_BBOB + ALL_CEC)
    if hasattr(f, 'f_global') or hasattr(f, 'x_global')
]


# =============================================================================
# Skip Markers
# =============================================================================

requires_sklearn = pytest.mark.skipif(
    not HAS_ML,
    reason="Requires scikit-learn: pip install surfaces[ml]"
)

requires_cec2013 = pytest.mark.skipif(
    not HAS_CEC2013,
    reason="Requires CEC 2013 data: pip install surfaces[cec]"
)

requires_cec2014 = pytest.mark.skipif(
    not HAS_CEC2014,
    reason="Requires CEC 2014 data: pip install surfaces[cec]"
)

requires_cec2017 = pytest.mark.skipif(
    not HAS_CEC2017,
    reason="Requires CEC 2017 data: pip install surfaces[cec]"
)

requires_cec = pytest.mark.skipif(
    not (HAS_CEC2013 or HAS_CEC2014 or HAS_CEC2017),
    reason="Requires CEC data: pip install surfaces[cec]"
)

requires_viz = pytest.mark.skipif(
    not HAS_VIZ,
    reason="Requires visualization deps: pip install surfaces[viz]"
)

requires_streamlit = pytest.mark.skipif(
    not HAS_STREAMLIT,
    reason="Requires streamlit: pip install surfaces[dashboard]"
)


# =============================================================================
# Helper Functions
# =============================================================================

def instantiate_function(func_class, n_dim=None):
    """Instantiate a test function with appropriate parameters.

    Parameters
    ----------
    func_class : type
        The test function class to instantiate.
    n_dim : int, optional
        Number of dimensions. If None, uses default or 2.

    Returns
    -------
    instance
        Instantiated test function.

    Raises
    ------
    pytest.skip
        If the function requires an optional dependency that is not installed.
    """
    try:
        # Check if function requires n_dim
        if n_dim is not None:
            return func_class(n_dim=n_dim)
        return func_class()
    except TypeError as e:
        # Function requires n_dim parameter
        if "n_dim" in str(e) or "required positional argument" in str(e):
            dim = n_dim if n_dim is not None else 2
            return func_class(n_dim=dim)
        raise
    except ImportError as e:
        # Function requires optional dependency
        pytest.skip(f"{func_class.__name__} requires optional dependency: {e}")


def get_sample_params(func):
    """Get sample parameters from a function's search space.

    Parameters
    ----------
    func : BaseTestFunction
        Instantiated test function.

    Returns
    -------
    dict
        Sample parameters (first value from each dimension).
    """
    search_space = func.search_space
    return {
        key: list(values)[0] if hasattr(values, "__iter__") else values
        for key, values in search_space.items()
    }


def get_middle_params(func):
    """Get middle-range parameters from a function's search space.

    Parameters
    ----------
    func : BaseTestFunction
        Instantiated test function.

    Returns
    -------
    dict
        Parameters at the middle of each dimension's range.
    """
    search_space = func.search_space
    params = {}
    for key, values in search_space.items():
        if hasattr(values, "__iter__"):
            values_list = list(values)
            mid_idx = len(values_list) // 2
            params[key] = values_list[mid_idx]
        else:
            params[key] = values
    return params


def func_id(func_class):
    """Generate a readable ID for a test function class."""
    return func_class.__name__


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(params=algebraic_functions, ids=func_id)
def algebraic_function(request):
    """Parametrized fixture for all algebraic functions."""
    return instantiate_function(request.param)


@pytest.fixture(params=algebraic_functions_1d, ids=func_id)
def algebraic_function_1d(request):
    """Parametrized fixture for 1D algebraic functions."""
    return instantiate_function(request.param)


@pytest.fixture(params=algebraic_functions_2d, ids=func_id)
def algebraic_function_2d(request):
    """Parametrized fixture for 2D algebraic functions."""
    return instantiate_function(request.param)


@pytest.fixture(params=algebraic_functions_nd, ids=func_id)
def algebraic_function_nd(request):
    """Parametrized fixture for N-D algebraic functions."""
    return instantiate_function(request.param, n_dim=2)


@pytest.fixture(params=engineering_functions, ids=func_id)
def engineering_function(request):
    """Parametrized fixture for engineering functions."""
    return instantiate_function(request.param)


@pytest.fixture(params=BBOB_FUNCTION_LIST, ids=func_id)
def bbob_function(request):
    """Parametrized fixture for BBOB functions."""
    return instantiate_function(request.param, n_dim=2)


@pytest.fixture
def quick_ml_params():
    """Minimal parameters for ML functions to speed up tests."""
    if not HAS_ML:
        pytest.skip("Requires scikit-learn")

    from surfaces.test_functions.machine_learning.tabular.classification.datasets import (
        iris_data,
    )
    return {"cv": 2, "dataset": iris_data}


@pytest.fixture
def quick_regression_params():
    """Minimal parameters for regression ML functions."""
    if not HAS_ML:
        pytest.skip("Requires scikit-learn")

    from surfaces.test_functions.machine_learning.tabular.regression.datasets import (
        diabetes_data,
    )
    return {"cv": 2, "dataset": diabetes_data}


# =============================================================================
# Pytest Hooks
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: Quick sanity checks (<1s each)")
    config.addinivalue_line("markers", "slow: Tests that take >5s")
    config.addinivalue_line("markers", "ml: Machine learning functions (require sklearn)")
    config.addinivalue_line("markers", "cec: CEC competition functions")
    config.addinivalue_line("markers", "cec2013: CEC 2013 functions")
    config.addinivalue_line("markers", "cec2014: CEC 2014 functions")
    config.addinivalue_line("markers", "cec2017: CEC 2017 functions")
    config.addinivalue_line("markers", "bbob: BBOB benchmark functions")
    config.addinivalue_line("markers", "algebraic: Algebraic/mathematical functions")
    config.addinivalue_line("markers", "engineering: Engineering design functions")
    config.addinivalue_line("markers", "requires_data: Requires external data files")
    config.addinivalue_line("markers", "viz: Visualization tests (require plotly/matplotlib)")
    config.addinivalue_line("markers", "dashboard: Streamlit dashboard tests")


def pytest_collection_modifyitems(config, items):
    """Auto-apply markers based on test location and name."""
    for item in items:
        # Apply markers based on test file path
        test_path = str(item.fspath)

        if "test_ml" in test_path or "machine_learning" in test_path:
            item.add_marker(pytest.mark.ml)

        if "cec2013" in test_path:
            item.add_marker(pytest.mark.cec2013)
            item.add_marker(pytest.mark.cec)

        if "cec2014" in test_path:
            item.add_marker(pytest.mark.cec2014)
            item.add_marker(pytest.mark.cec)

        if "cec2017" in test_path:
            item.add_marker(pytest.mark.cec2017)
            item.add_marker(pytest.mark.cec)

        if "bbob" in test_path:
            item.add_marker(pytest.mark.bbob)

        if "algebraic" in test_path:
            item.add_marker(pytest.mark.algebraic)

        if "engineering" in test_path:
            item.add_marker(pytest.mark.engineering)

        if "smoke" in test_path:
            item.add_marker(pytest.mark.smoke)

        if "visualization" in test_path:
            item.add_marker(pytest.mark.viz)

        if "dashboard" in test_path:
            item.add_marker(pytest.mark.dashboard)

        if "test_optimization" in test_path:
            item.add_marker(pytest.mark.slow)
