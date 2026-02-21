# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Interface Compliance Tests for Test Functions
=============================================

This module automatically discovers test function classes and validates
they follow the required interface contract. No manual imports needed -
new test functions are automatically included.

All tests in this module are STATIC - they inspect class definitions
without instantiating objects. This allows testing expensive ML functions
that require heavy dependencies (TensorFlow, PyTorch, gymnasium) without
triggering imports or model training.

Test Categories
---------------
| Category               | Instantiates? | What it Tests                        |
|------------------------|---------------|--------------------------------------|
| TestDiscovery          | No            | Discovery mechanism itself works     |
| TestClassAttributes    | No            | Static class-level attributes        |
| TestStaticInterface    | No            | Required methods/properties on class |
| TestStaticSpecification| No            | _spec dict on class hierarchy        |

Instantiation tests (creating instances, calling functions) are in:
    tests/full/smoke/test_instantiation.py

Required Interface (every concrete test function MUST have)
-----------------------------------------------------------
- `name` or `_spec["name"]`: Human-readable function name
- `_spec`: Dict with function characteristics (continuous, differentiable, etc.)
- `_create_objective_function()`: Method that creates the objective function
- `search_space` property or `para_names`/`n_dim` to define parameter space
- `tagline`: Short description of the function
- `reference_url`: URL to documentation or paper

Usage
-----
Run all tests:
    pytest tests/core/properties/test_interface_compliance.py -v

Run only static checks (all tests are static now):
    pytest tests/core/properties/test_interface_compliance.py -m static -v
"""

import importlib
import inspect
import pkgutil
from typing import Any, Dict, List, Set, Type

import pytest

from surfaces.test_functions._base_test_function import BaseTestFunction

# =============================================================================
# Discovery Infrastructure
# =============================================================================

# Base classes that should be excluded from testing (abstract/intermediate)
BASE_CLASS_NAMES = {
    "BaseTestFunction",
    "AlgebraicFunction",
    "MathematicalFunction",
    "BBOBFunction",
    "CECFunction",
    "CEC2013Function",
    "CEC2014Function",
    "CEC2017Function",
    "EngineeringFunction",
    "MachineLearningFunction",
    "SimulationFunction",
    "ODESimulationFunction",
    "MultiObjectiveFunction",
    # ML base classes
    "BaseTabular",
    "BaseClassification",
    "BaseRegression",
    "BaseForecaster",
    "BaseTSClassifier",
    "BaseImageClassification",
    "BaseNeuralArchitectureSearch",
    "BaseNASFunction",
    "BaseTransferLearning",
    "BaseDataAugmentation",
    "BaseReinforcementLearning",
    "BaseFeatureEngineering",
    "BaseEnsembleOptimization",
    "BasePipeline",
    "BaseLLMOptimization",
    "BaseTabularFeatureEngineering",
    "BaseTabularEnsemble",
    "BaseTabularPipeline",
    "BaseTimeSeries",
    "BaseImage",
}


def is_base_class(cls: Type) -> bool:
    """Check if a class is a base/abstract class that should be excluded."""
    name = cls.__name__
    if name in BASE_CLASS_NAMES:
        return True
    if name.startswith("Base"):
        return True
    if name.startswith("_"):
        return True
    return False


def is_concrete_test_function(cls: Type) -> bool:
    """Check if a class is a concrete test function implementation."""
    if not inspect.isclass(cls):
        return False
    if not issubclass(cls, BaseTestFunction):
        return False
    if cls is BaseTestFunction:
        return False
    if is_base_class(cls):
        return False
    if not hasattr(cls, "_create_objective_function"):
        return False
    # Check if method is actually implemented (not just NotImplementedError)
    try:
        method = getattr(cls, "_create_objective_function")
        source = inspect.getsource(method)
        if "NotImplementedError" in source and "raise" in source:
            return False
    except (TypeError, OSError):
        pass
    return True


def discover_test_function_classes() -> List[Type[BaseTestFunction]]:
    """Automatically discover all concrete test function classes.

    Recursively scans surfaces.test_functions and finds all classes
    that inherit from BaseTestFunction and are not abstract.

    This function discovers ALL test function classes including expensive
    ML functions. Since tests are static (no instantiation), this is safe.
    """
    import surfaces.test_functions

    discovered: Set[Type[BaseTestFunction]] = set()

    def should_skip_module(modname: str) -> bool:
        """Check if module should be skipped."""
        if "._" in modname or modname.endswith("_"):
            return True
        return False

    def scan_module(module) -> None:
        for name in dir(module):
            obj = getattr(module, name)
            if is_concrete_test_function(obj):
                discovered.add(obj)

        if hasattr(module, "__path__"):
            for _, modname, _ in pkgutil.walk_packages(
                module.__path__, prefix=module.__name__ + "."
            ):
                if should_skip_module(modname):
                    continue
                try:
                    submodule = importlib.import_module(modname)
                    scan_module(submodule)
                except (ImportError, Exception):
                    pass

    scan_module(surfaces.test_functions)
    return sorted(discovered, key=lambda x: x.__name__)


def class_id(func_class: Type[BaseTestFunction]) -> str:
    """Generate readable test ID: ClassName[category]."""
    module = func_class.__module__
    if "bbob" in module:
        suffix = "bbob"
    elif "cec2017" in module:
        suffix = "cec2017"
    elif "cec2014" in module:
        suffix = "cec2014"
    elif "cec2013" in module:
        suffix = "cec2013"
    elif "algebraic" in module:
        suffix = "algebraic"
    elif "neural_architecture_search" in module:
        suffix = "nas"
    elif "reinforcement_learning" in module:
        suffix = "rl"
    elif "transfer_learning" in module:
        suffix = "tl"
    elif "data_augmentation" in module:
        suffix = "aug"
    elif "llm_optimization" in module:
        suffix = "llm"
    elif "pipelines" in module:
        suffix = "pipe"
    elif "machine_learning" in module:
        suffix = "ml"
    elif "simulation" in module:
        suffix = "sim"
    elif "constrained" in module:
        suffix = "constrained"
    else:
        suffix = module.split(".")[-1]
    return f"{func_class.__name__}[{suffix}]"


def _get_merged_spec(cls: Type) -> Dict[str, Any]:
    """Merge _spec dicts from class hierarchy without instantiation.

    Traverses the MRO (Method Resolution Order) and merges all _spec
    dicts from base classes to the concrete class. Later classes in
    the hierarchy override earlier ones.

    Parameters
    ----------
    cls : Type
        The class to get merged spec for.

    Returns
    -------
    Dict[str, Any]
        Merged spec dict from entire class hierarchy.
    """
    merged: Dict[str, Any] = {}
    for klass in reversed(cls.__mro__):
        if hasattr(klass, "_spec") and isinstance(klass._spec, dict):
            merged.update(klass._spec)
    return merged


# Discover all test function classes once at module load time
ALL_TEST_FUNCTION_CLASSES = discover_test_function_classes()


@pytest.mark.static
@pytest.mark.parametrize("func_class", ALL_TEST_FUNCTION_CLASSES, ids=class_id)
class TestClassAttributes:
    """Test static class-level attributes without instantiation.

    These tests are fast because they only inspect the class definition.
    They verify that required class attributes exist before any instance
    is created.

    Why these matter:
    - `name`: Used for display, documentation, and function identification
    - `_spec`: Defines function characteristics (convex, separable, etc.)
    - `_create_objective_function`: The method that defines the actual function
    """

    def test_has_name(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must have a human-readable name."""
        assert hasattr(func_class, "name") and isinstance(
            func_class.name, str
        ), f"{func_class.__name__}: Missing 'name' attribute"

    def test_has_spec_dict(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must have _spec dict defining function characteristics."""
        assert hasattr(func_class, "_spec"), f"{func_class.__name__}: Missing '_spec'"
        assert isinstance(func_class._spec, dict), f"{func_class.__name__}: _spec must be dict"

    def test_has_create_objective_method(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must have _create_objective_function method."""
        assert hasattr(
            func_class, "_create_objective_function"
        ), f"{func_class.__name__}: Missing '_create_objective_function' method"


@pytest.mark.static
@pytest.mark.parametrize("func_class", ALL_TEST_FUNCTION_CLASSES, ids=class_id)
class TestStaticInterface:
    """Test static interface requirements without instantiation.

    These tests verify that classes have the required properties and methods
    defined at the class level, without creating instances.

    Why these matter:
    - `search_space` or parameter definition: Optimizers need to know what to optimize
    - `__call__`: Functions must be callable (inherited from base)
    - `tagline`: Documentation requires a short description
    - `reference_url`: Users need to find more information
    """

    def test_has_search_space_or_para_names(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must define parameter space via search_space, para_names, or n_dim.

        ML functions use para_names with *_default attributes.
        Algebraic functions use n_dim with default_bounds.
        """
        has_search_space = hasattr(func_class, "search_space")
        has_para_names = hasattr(func_class, "para_names") and func_class.para_names
        has_n_dim = hasattr(func_class, "n_dim") and func_class.n_dim is not None

        assert (
            has_search_space or has_para_names or has_n_dim
        ), f"{func_class.__name__}: Must have 'search_space', 'para_names', or 'n_dim'"

    def test_is_callable_class(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must have __call__ method (inherited from BaseTestFunction)."""
        assert hasattr(
            func_class, "__call__"
        ), f"{func_class.__name__}: Must have '__call__' method"

    def test_has_tagline(self, func_class: Type[BaseTestFunction]) -> None:
        """Class should have tagline for documentation.

        Note: This is currently a soft requirement - we warn but don't fail
        for classes missing tagline, as many existing functions need updates.
        """
        has_tagline = hasattr(func_class, "tagline") and func_class.tagline
        has_tagline_in_spec = (
            hasattr(func_class, "_spec")
            and isinstance(func_class._spec, dict)
            and "tagline" in func_class._spec
        )
        merged_spec = _get_merged_spec(func_class)
        has_tagline_merged = "tagline" in merged_spec

        if not (has_tagline or has_tagline_in_spec or has_tagline_merged):
            pytest.skip(f"{func_class.__name__}: Missing 'tagline' (soft requirement)")

    def test_has_reference_url(self, func_class: Type[BaseTestFunction]) -> None:
        """Class should have reference_url for documentation.

        Note: This is currently a soft requirement - we warn but don't fail
        for classes missing reference_url, as many existing functions need updates.
        """
        has_ref = hasattr(func_class, "reference_url") and func_class.reference_url
        has_ref_in_spec = (
            hasattr(func_class, "_spec")
            and isinstance(func_class._spec, dict)
            and "reference_url" in func_class._spec
        )
        merged_spec = _get_merged_spec(func_class)
        has_ref_merged = "reference_url" in merged_spec

        if not (has_ref or has_ref_in_spec or has_ref_merged):
            pytest.skip(f"{func_class.__name__}: Missing 'reference_url' (soft requirement)")


@pytest.mark.static
@pytest.mark.parametrize("func_class", ALL_TEST_FUNCTION_CLASSES, ids=class_id)
class TestStaticSpecification:
    """Test the _spec specification system without instantiation.

    The spec system defines function characteristics that help users
    choose appropriate test functions for their experiments.

    These tests use _get_merged_spec() to check the merged spec from
    the entire class hierarchy, similar to what the spec property does
    at runtime.

    Why these matter:
    - Users filter functions by properties (e.g., "give me all unimodal functions")
    - Documentation generation uses these values
    - Some optimizers behave differently based on properties
    """

    def test_spec_is_dict(self, func_class: Type[BaseTestFunction]) -> None:
        """Class _spec must be a dict."""
        assert isinstance(func_class._spec, dict), f"{func_class.__name__}: _spec must be dict"

    def test_merged_spec_has_required_keys(self, func_class: Type[BaseTestFunction]) -> None:
        """Merged spec must contain: continuous, differentiable, default_bounds.

        These can come from the class itself or any parent in the hierarchy.
        Engineering functions may use variable_bounds instead of default_bounds.
        """
        merged = _get_merged_spec(func_class)
        required = {"continuous", "differentiable"}

        # Check for bounds - either default_bounds or variable_bounds
        has_default_bounds = "default_bounds" in merged
        has_variable_bounds = hasattr(func_class, "variable_bounds")

        missing = required - set(merged.keys())
        assert not missing, f"{func_class.__name__}: merged spec missing keys: {missing}"

        assert (
            has_default_bounds or has_variable_bounds
        ), f"{func_class.__name__}: merged spec missing 'default_bounds' or 'variable_bounds'"

    def test_merged_spec_continuous_is_bool(self, func_class: Type[BaseTestFunction]) -> None:
        """Merged spec['continuous'] must be boolean."""
        merged = _get_merged_spec(func_class)
        val = merged.get("continuous")
        assert isinstance(
            val, bool
        ), f"{func_class.__name__}: spec['continuous'] must be bool, got {type(val).__name__}"

    def test_merged_spec_differentiable_is_bool(self, func_class: Type[BaseTestFunction]) -> None:
        """Merged spec['differentiable'] must be boolean."""
        merged = _get_merged_spec(func_class)
        val = merged.get("differentiable")
        assert isinstance(
            val, bool
        ), f"{func_class.__name__}: spec['differentiable'] must be bool, got {type(val).__name__}"

    def test_merged_spec_default_bounds_valid(self, func_class: Type[BaseTestFunction]) -> None:
        """Merged spec['default_bounds'] must be (min, max) tuple with min < max.

        Engineering functions with variable_bounds are exempt since they have
        per-variable bounds instead of a single default_bounds tuple.
        """
        merged = _get_merged_spec(func_class)
        bounds = merged.get("default_bounds")

        # Engineering functions use variable_bounds instead of default_bounds
        if bounds is None and hasattr(func_class, "variable_bounds"):
            # variable_bounds validation would happen in instantiation tests
            return

        if bounds is None:
            pytest.fail(f"{func_class.__name__}: missing default_bounds in merged spec")

        assert isinstance(
            bounds, (tuple, list)
        ), f"{func_class.__name__}: default_bounds must be tuple/list"
        assert len(bounds) == 2, f"{func_class.__name__}: default_bounds must have 2 elements"
        assert (
            bounds[0] < bounds[1]
        ), f"{func_class.__name__}: default_bounds[0] must be < default_bounds[1]"
