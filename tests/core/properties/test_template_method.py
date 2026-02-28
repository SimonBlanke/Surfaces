# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Template Method Pattern Compliance Tests
=========================================

These tests verify that every discovered test function class provides
the required override points for the template method pattern:

    All classes:
        _objective(params)          [required, via self or intermediate base]
        _default_search_space()     [required, via self or intermediate base]

    EngineeringFunction subclasses:
        _raw_objective(params)      [required]
        _constraints(params)        [optional, default returns []]

    MachineLearningFunction subclasses:
        _ml_objective(params)       [required]

Additionally verifies that no class still uses the legacy closure-based
pattern (_create_objective_function).

Usage:
    pytest tests/core/properties/test_template_method.py -v
"""

from typing import Type

import pytest

from surfaces.test_functions._base_test_function import BaseTestFunction

from .test_interface_compliance import (
    ALL_TEST_FUNCTION_CLASSES,
    class_id,
)


def _has_method_in_mro(cls: Type, method_name: str, stop_at: Type = BaseTestFunction) -> bool:
    """Check if method_name is defined anywhere in the MRO between cls and stop_at (exclusive)."""
    for klass in cls.__mro__:
        if klass is stop_at:
            break
        if method_name in klass.__dict__:
            return True
    return False


def _is_engineering_subclass(cls: Type) -> bool:
    """Check if cls inherits from EngineeringFunction."""
    from surfaces.test_functions.algebraic.constrained._base_engineering_function import (
        EngineeringFunction,
    )

    return issubclass(cls, EngineeringFunction) and cls is not EngineeringFunction


def _is_ml_subclass(cls: Type) -> bool:
    """Check if cls inherits from MachineLearningFunction."""
    from surfaces.test_functions.machine_learning._base_machine_learning import (
        MachineLearningFunction,
    )

    return issubclass(cls, MachineLearningFunction) and cls is not MachineLearningFunction


@pytest.mark.static
@pytest.mark.parametrize("func_class", ALL_TEST_FUNCTION_CLASSES, ids=class_id)
class TestTemplateMethodCompliance:
    """Verify every class provides the required override points."""

    def test_has_objective_method(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must provide _objective (directly or via intermediate base)."""
        assert _has_method_in_mro(func_class, "_objective"), (
            f"{func_class.__name__}: does not override _objective. "
            f"Either implement _objective directly or ensure an intermediate "
            f"base class provides a sub-template."
        )

    def test_has_default_search_space(self, func_class: Type[BaseTestFunction]) -> None:
        """Class must provide _default_search_space (directly or via intermediate base)."""
        assert _has_method_in_mro(func_class, "_default_search_space"), (
            f"{func_class.__name__}: does not provide _default_search_space. "
            f"Implement _default_search_space() to define the parameter space."
        )

    def test_no_search_space_property_override(self, func_class: Type[BaseTestFunction]) -> None:
        """Subclasses must not override search_space as a property.

        search_space is a fixed property on BaseTestFunction that delegates
        to _default_search_space(). Overriding it breaks the template pattern.
        """
        for klass in func_class.__mro__:
            if klass is BaseTestFunction:
                break
            if "search_space" in klass.__dict__:
                assert False, (
                    f"{func_class.__name__}: overrides search_space directly "
                    f"(in {klass.__name__}). Override _default_search_space() instead."
                )

    def test_no_create_objective_function(self, func_class: Type[BaseTestFunction]) -> None:
        """No class should still use the legacy closure-based pattern."""
        assert not _has_method_in_mro(func_class, "_create_objective_function"), (
            f"{func_class.__name__}: still defines _create_objective_function. "
            f"Replace with _objective method pattern."
        )


_ENGINEERING_CLASSES = [c for c in ALL_TEST_FUNCTION_CLASSES if _is_engineering_subclass(c)]


@pytest.mark.static
@pytest.mark.parametrize("func_class", _ENGINEERING_CLASSES, ids=class_id)
class TestEngineeringTemplateMethod:
    """Verify engineering subclasses provide _raw_objective."""

    def test_has_raw_objective(self, func_class: Type[BaseTestFunction]) -> None:
        """Engineering subclass must provide _raw_objective."""
        from surfaces.test_functions.algebraic.constrained._base_engineering_function import (
            EngineeringFunction,
        )

        assert _has_method_in_mro(func_class, "_raw_objective", stop_at=EngineeringFunction), (
            f"{func_class.__name__}: does not implement _raw_objective. "
            f"Engineering subclasses must define _raw_objective(self, params)."
        )


_ML_CLASSES = [c for c in ALL_TEST_FUNCTION_CLASSES if _is_ml_subclass(c)]


@pytest.mark.static
@pytest.mark.parametrize("func_class", _ML_CLASSES, ids=class_id)
class TestMLTemplateMethod:
    """Verify ML subclasses provide _ml_objective."""

    def test_has_ml_objective(self, func_class: Type[BaseTestFunction]) -> None:
        """ML subclass must provide _ml_objective."""
        from surfaces.test_functions.machine_learning._base_machine_learning import (
            MachineLearningFunction,
        )

        assert _has_method_in_mro(func_class, "_ml_objective", stop_at=MachineLearningFunction), (
            f"{func_class.__name__}: does not implement _ml_objective. "
            f"ML subclasses must define _ml_objective(self, params)."
        )
