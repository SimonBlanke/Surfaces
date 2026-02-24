# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Template Method Migration Tracking Tests
=========================================

These tests track the migration from the closure-based pattern
(_create_objective_function setting self.pure_objective_function)
to the template method pattern (_objective / _raw_objective / _ml_objective).

The _KNOWN_MIGRATION_PENDING set contains all classes not yet migrated.
As migration proceeds, classes are removed from this set. When the set
is empty, the migration is complete and Phase 6 (remove backward compat)
can proceed.

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

# =============================================================================
# Known migration-pending classes (module.ClassName)
# Remove entries as each class is migrated to _objective pattern.
# When this set is empty, Phase 6 (remove backward compat) can proceed.
# =============================================================================

_KNOWN_MIGRATION_PENDING = frozenset()  # Migration complete


def _class_key(cls: Type) -> str:
    """Unique key for a class: module.ClassName."""
    return f"{cls.__module__}.{cls.__name__}"


def _is_migration_pending(cls: Type) -> bool:
    """Check if a class is in the known migration-pending set."""
    return _class_key(cls) in _KNOWN_MIGRATION_PENDING


def _has_own_objective(cls: Type) -> bool:
    """Check if the class (or an intermediate base above BaseTestFunction)
    provides a non-fallback _objective method.

    A class passes if _objective is overridden anywhere in the MRO between
    the concrete class and BaseTestFunction (exclusive).
    """
    for klass in cls.__mro__:
        if klass is BaseTestFunction:
            break
        if "_objective" in klass.__dict__:
            return True
    return False


def _has_own_create_objective(cls: Type) -> bool:
    """Check if the class defines _create_objective_function in its own __dict__
    or in an intermediate base above BaseTestFunction."""
    for klass in cls.__mro__:
        if klass is BaseTestFunction:
            break
        if "_create_objective_function" in klass.__dict__:
            return True
    return False


# =============================================================================
# Tests
# =============================================================================


@pytest.mark.static
@pytest.mark.parametrize("func_class", ALL_TEST_FUNCTION_CLASSES, ids=class_id)
class TestTemplateMethodMigration:
    """Track migration from closure pattern to template method pattern."""

    def test_has_objective_method(self, func_class: Type[BaseTestFunction]) -> None:
        """Class should provide _objective (directly or via intermediate base).

        Migrated classes override _objective somewhere in the hierarchy above
        BaseTestFunction. Classes still in _KNOWN_MIGRATION_PENDING are allowed
        to fail this check (they use the backward-compat fallback).
        """
        if _is_migration_pending(func_class):
            pytest.skip(f"{func_class.__name__}: migration pending")

        assert _has_own_objective(func_class), (
            f"{func_class.__name__}: does not override _objective. "
            f"Either implement _objective directly or ensure an intermediate "
            f"base class provides a sub-template."
        )

    def test_no_create_objective_function(self, func_class: Type[BaseTestFunction]) -> None:
        """Migrated classes should not define _create_objective_function.

        After migration, the closure-based pattern should be replaced with
        the _objective method. Classes still in _KNOWN_MIGRATION_PENDING are
        allowed to still have _create_objective_function.
        """
        if _is_migration_pending(func_class):
            pytest.skip(f"{func_class.__name__}: migration pending")

        assert not _has_own_create_objective(func_class), (
            f"{func_class.__name__}: still defines _create_objective_function. "
            f"Replace with _objective method pattern."
        )


@pytest.mark.static
class TestMigrationProgress:
    """Track overall migration progress."""

    def test_pending_set_not_stale(self) -> None:
        """All entries in _KNOWN_MIGRATION_PENDING must correspond to
        discovered test function classes. Catches typos and removed classes."""
        discovered_keys = {_class_key(cls) for cls in ALL_TEST_FUNCTION_CLASSES}
        stale = _KNOWN_MIGRATION_PENDING - discovered_keys
        assert not stale, (
            f"Stale entries in _KNOWN_MIGRATION_PENDING (not found in discovery): "
            f"{sorted(stale)}"
        )

    def test_migration_count(self) -> None:
        """Report migration progress (informational)."""
        total = len(ALL_TEST_FUNCTION_CLASSES)
        pending = sum(1 for cls in ALL_TEST_FUNCTION_CLASSES if _is_migration_pending(cls))
        migrated = total - pending
        print(f"\nMigration progress: {migrated}/{total} classes migrated " f"({pending} pending)")
