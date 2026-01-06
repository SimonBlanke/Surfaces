"""Shared utilities for Surfaces documentation generators.

This module provides common functions for extracting metadata from test
function classes and organizing them by category.

Usage
-----
>>> from docs._generators import get_all_test_functions, extract_metadata
>>> categories = get_all_test_functions()
>>> for name, funcs in categories.items():
...     print(f"{name}: {len(funcs)} functions")
"""

import inspect
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

# Add src to path for imports
from .config import SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def get_all_test_functions() -> Dict[str, List[Type]]:
    """
    Get all test functions organized by category.

    Returns
    -------
    dict
        Dictionary with category names as keys and lists of function classes
        as values. Categories include:
        - algebraic_1d, algebraic_2d, algebraic_nd
        - ml_tabular_classification, ml_tabular_regression
        - ml_image_classification
        - ml_timeseries_classification, ml_timeseries_forecasting
        - engineering

    Examples
    --------
    >>> categories = get_all_test_functions()
    >>> len(categories['algebraic_2d'])
    18
    """
    categories = {}

    # Algebraic functions (always available)
    try:
        from surfaces.test_functions.algebraic import (
            algebraic_functions_1d,
            algebraic_functions_2d,
            algebraic_functions_nd,
        )

        categories["algebraic_1d"] = list(algebraic_functions_1d)
        categories["algebraic_2d"] = list(algebraic_functions_2d)
        categories["algebraic_nd"] = list(algebraic_functions_nd)
    except ImportError as e:
        print(f"Warning: Could not import algebraic functions: {e}")
        categories["algebraic_1d"] = []
        categories["algebraic_2d"] = []
        categories["algebraic_nd"] = []

    # Constrained/Engineering functions (always available)
    try:
        from surfaces.test_functions.algebraic.constrained import constrained_functions

        categories["engineering"] = list(constrained_functions)
    except ImportError as e:
        print(f"Warning: Could not import engineering functions: {e}")
        categories["engineering"] = []

    # Machine learning functions (require sklearn)
    try:
        from surfaces.test_functions.machine_learning import machine_learning_functions

        # Categorize ML functions by type
        ml_tabular_clf = []
        ml_tabular_reg = []
        ml_image_clf = []
        ml_ts_clf = []
        ml_ts_forecast = []

        for func_class in machine_learning_functions:
            name = func_class.__name__.lower()
            module = func_class.__module__.lower()

            if "image" in module:
                ml_image_clf.append(func_class)
            elif "timeseries" in module or "time_series" in module:
                if "classifier" in name or "classification" in module:
                    ml_ts_clf.append(func_class)
                else:
                    ml_ts_forecast.append(func_class)
            elif "classifier" in name:
                ml_tabular_clf.append(func_class)
            elif "regressor" in name:
                ml_tabular_reg.append(func_class)

        categories["ml_tabular_classification"] = ml_tabular_clf
        categories["ml_tabular_regression"] = ml_tabular_reg
        categories["ml_image_classification"] = ml_image_clf
        categories["ml_timeseries_classification"] = ml_ts_clf
        categories["ml_timeseries_forecasting"] = ml_ts_forecast

    except ImportError:
        # sklearn not installed
        categories["ml_tabular_classification"] = []
        categories["ml_tabular_regression"] = []
        categories["ml_image_classification"] = []
        categories["ml_timeseries_classification"] = []
        categories["ml_timeseries_forecasting"] = []

    # BBOB functions (if available)
    try:
        from surfaces.test_functions.benchmark.bbob import (
            high_conditioning,
            low_conditioning,
            multimodal_adequate,
            multimodal_weak,
            separable,
        )
        from surfaces.test_functions.benchmark.bbob._base_bbob import BBOBFunction

        bbob_funcs = []
        for module in [
            separable,
            low_conditioning,
            high_conditioning,
            multimodal_adequate,
            multimodal_weak,
        ]:
            # Find classes that are subclasses of BBOBFunction
            for name in dir(module):
                if name.startswith("_"):
                    continue
                cls = getattr(module, name, None)
                if (
                    cls is not None
                    and inspect.isclass(cls)
                    and issubclass(cls, BBOBFunction)
                    and cls is not BBOBFunction
                ):
                    bbob_funcs.append(cls)
        categories["bbob"] = bbob_funcs
    except ImportError:
        categories["bbob"] = []

    # CEC functions (if available)
    try:
        from surfaces.test_functions.benchmark import cec

        cec_funcs = []
        for submodule_name in ["cec2014", "cec2017"]:
            submodule = getattr(cec, submodule_name, None)
            if submodule and hasattr(submodule, "__all__"):
                for name in submodule.__all__:
                    cls = getattr(submodule, name, None)
                    if cls is not None and inspect.isclass(cls):
                        cec_funcs.append(cls)
        categories["cec"] = cec_funcs
    except ImportError:
        categories["cec"] = []

    return categories


def extract_metadata(func_class: Type) -> Dict[str, Any]:
    """
    Extract documentation metadata from a test function class.

    Parameters
    ----------
    func_class : type
        A test function class (subclass of BaseTestFunction).

    Returns
    -------
    dict
        Metadata dictionary with the following keys:
        - name: str - Class name (e.g., "SphereFunction")
        - display_name: str - Human-readable name (e.g., "Sphere Function")
        - internal_name: str - Snake case name (e.g., "sphere_function")
        - n_dim: int or str - Number of dimensions, or "N" if scalable
        - default_bounds: tuple - (min, max) bounds for parameters
        - f_global: float or None - Global minimum value
        - x_global: array or None - Global minimum location
        - docstring: str - Full docstring
        - first_line: str - First line of docstring (short description)
        - has_latex: bool - Whether latex_formula attribute exists
        - latex_formula: str or None - LaTeX formula if available
        - module: str - Full module path

    Examples
    --------
    >>> from surfaces.test_functions import SphereFunction
    >>> meta = extract_metadata(SphereFunction)
    >>> meta['name']
    'SphereFunction'
    >>> meta['n_dim']
    'N'
    """
    # Get docstring info
    doc = func_class.__doc__ or ""
    first_line = ""
    if doc:
        # Get first non-empty line
        for line in doc.split("\n"):
            stripped = line.strip()
            if stripped:
                first_line = stripped
                break

    # Determine n_dim
    n_dim = getattr(func_class, "n_dim", None)
    if n_dim is None:
        # Check if it's scalable (takes n_dim in __init__)
        try:
            sig = inspect.signature(func_class.__init__)
            if "n_dim" in sig.parameters:
                n_dim = "N"
        except (ValueError, TypeError):
            pass

    # Get default bounds
    default_bounds = getattr(func_class, "default_bounds", (-5.0, 5.0))

    # Handle _spec attribute for bounds if default_bounds not set
    if hasattr(func_class, "_spec") and "default_bounds" in func_class._spec:
        default_bounds = func_class._spec["default_bounds"]

    # Get global optimum info
    f_global = getattr(func_class, "f_global", None)
    x_global = getattr(func_class, "x_global", None)

    # Get display name
    display_name = getattr(func_class, "name", None)
    if display_name is None:
        # Convert class name to readable form
        display_name = func_class.__name__
        # Remove "Function" suffix if present
        if display_name.endswith("Function"):
            display_name = display_name[:-8]
        # Add spaces before capitals
        import re

        display_name = re.sub(r"(?<!^)(?=[A-Z])", " ", display_name)

    # Get internal name
    internal_name = getattr(func_class, "_name_", None)
    if internal_name is None:
        # Convert class name to snake_case
        import re

        internal_name = re.sub(r"(?<!^)(?=[A-Z])", "_", func_class.__name__).lower()

    return {
        "name": func_class.__name__,
        "display_name": display_name,
        "internal_name": internal_name,
        "n_dim": n_dim,
        "default_bounds": default_bounds,
        "f_global": f_global,
        "x_global": x_global,
        "docstring": doc,
        "first_line": first_line,
        "has_latex": hasattr(func_class, "latex_formula"),
        "latex_formula": getattr(func_class, "latex_formula", None),
        "module": func_class.__module__,
    }


def format_value(value: Any, precision: int = 4) -> str:
    """
    Format a value for display in RST tables.

    Parameters
    ----------
    value : any
        Value to format (float, int, tuple, None, etc.)
    precision : int
        Number of significant digits for floats.

    Returns
    -------
    str
        Formatted string suitable for RST display.

    Examples
    --------
    >>> format_value(None)
    '---'
    >>> format_value(0.0)
    '0'
    >>> format_value(3.14159265)
    '3.142'
    >>> format_value((1.0, 2.0))
    '(1, 2)'
    """
    if value is None:
        return "---"

    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) < 0.0001 or abs(value) > 10000:
            return f"{value:.{precision}g}"
        return f"{value:.{precision}g}"

    if isinstance(value, (list, tuple)):
        formatted = ", ".join(format_value(v, precision) for v in value)
        return f"({formatted})"

    if hasattr(value, "__iter__") and not isinstance(value, str):
        # numpy arrays, etc.
        try:
            if len(value) <= 4:
                formatted = ", ".join(format_value(float(v), precision) for v in value)
                return f"({formatted})"
            else:
                return f"({len(value)}-dim)"
        except (TypeError, ValueError):
            return str(value)

    return str(value)


def get_function_hash(func_class: Type) -> str:
    """
    Get a hash of a function class for cache invalidation.

    The hash is based on the source code of the class, so it changes
    when the implementation changes.

    Parameters
    ----------
    func_class : type
        A test function class.

    Returns
    -------
    str
        8-character hexadecimal hash string.
    """
    import hashlib

    try:
        source = inspect.getsource(func_class)
        return hashlib.md5(source.encode()).hexdigest()[:8]
    except (OSError, TypeError):
        # Can't get source (built-in, etc.)
        return hashlib.md5(func_class.__name__.encode()).hexdigest()[:8]


def count_by_category(categories: Optional[Dict[str, List[Type]]] = None) -> Dict[str, int]:
    """
    Count functions in each category.

    Parameters
    ----------
    categories : dict, optional
        Category dictionary from get_all_test_functions().
        If None, will call get_all_test_functions().

    Returns
    -------
    dict
        Dictionary mapping category names to counts.

    Examples
    --------
    >>> counts = count_by_category()
    >>> counts['algebraic_2d']
    18
    """
    if categories is None:
        categories = get_all_test_functions()

    return {name: len(funcs) for name, funcs in categories.items()}


def get_total_count(categories: Optional[Dict[str, List[Type]]] = None) -> int:
    """
    Get total count of all test functions.

    Parameters
    ----------
    categories : dict, optional
        Category dictionary from get_all_test_functions().

    Returns
    -------
    int
        Total number of test functions across all categories.
    """
    counts = count_by_category(categories)
    return sum(counts.values())
