"""Registry and export chain consistency tests for ML functions.

Validates that ML function classes are properly registered and exported
through the entire __init__.py hierarchy, from concrete implementations
up to the top-level machine_learning module.

What is checked
---------------
| Test Class                  | What it catches                          |
|-----------------------------|------------------------------------------|
| TestMLRegistryCompleteness  | Class exists on disk but not registered  |
| TestMLExportConsistency     | __all__ and machine_learning_functions   |
|                             | out of sync                              |
| TestMLInitChainConsistency  | Class exported at child level but not    |
|                             | propagated to parent __init__.py         |

PR #17 scenario: A new classifier file was added and partially wired
into the __init__.py chain. These tests would catch every missed level.
"""

import importlib

import pytest

from surfaces.test_functions.machine_learning import (
    __all__ as ml_all,
)
from surfaces.test_functions.machine_learning import (
    machine_learning_functions,
)
from tests.full.properties._ml_discovery import ML_CLASSES


@pytest.mark.ml
@pytest.mark.static
class TestMLRegistryCompleteness:
    """Every discoverable ML class must be in the machine_learning_functions registry."""

    def test_all_discovered_in_registry(self):
        """Classes found via module scan must appear in machine_learning_functions.

        Catches: new file added to test_functions/ directory but not
        registered in machine_learning/__init__.py.
        """
        registry_names = {cls.__name__ for cls in machine_learning_functions}
        discovered_names = {cls.__name__ for cls in ML_CLASSES}

        missing = discovered_names - registry_names
        assert not missing, (
            f"ML classes discovered via module scan but NOT in "
            f"machine_learning_functions registry: {sorted(missing)}\n"
            f"Add these to surfaces/test_functions/machine_learning/__init__.py"
        )

    def test_no_phantom_entries_in_registry(self):
        """Every entry in machine_learning_functions must be a discoverable class.

        Catches: class removed from code but still listed in registry.
        """
        registry_names = {cls.__name__ for cls in machine_learning_functions}
        discovered_names = {cls.__name__ for cls in ML_CLASSES}

        phantom = registry_names - discovered_names
        assert not phantom, (
            f"Classes in machine_learning_functions but NOT discoverable "
            f"via module scan: {sorted(phantom)}\n"
            f"These may have been removed or renamed."
        )


@pytest.mark.ml
@pytest.mark.static
class TestMLExportConsistency:
    """__all__ and machine_learning_functions must stay in sync."""

    def test_all_matches_registry(self):
        """__all__ names must match machine_learning_functions class names."""
        all_names = set(ml_all)
        registry_names = {cls.__name__ for cls in machine_learning_functions}

        missing_in_all = registry_names - all_names
        missing_in_registry = all_names - registry_names

        errors = []
        if missing_in_all:
            errors.append(
                f"In machine_learning_functions but NOT in __all__: {sorted(missing_in_all)}"
            )
        if missing_in_registry:
            errors.append(
                f"In __all__ but NOT in machine_learning_functions: {sorted(missing_in_registry)}"
            )

        assert not errors, "\n".join(errors)

    def test_all_entries_importable(self):
        """Every name in __all__ must be accessible on the module."""
        import surfaces.test_functions.machine_learning as ml_module

        not_importable = []
        for name in ml_all:
            if not hasattr(ml_module, name):
                not_importable.append(name)

        assert not not_importable, (
            f"Names in __all__ but not importable from "
            f"surfaces.test_functions.machine_learning: {not_importable}"
        )


_HP = "surfaces.test_functions.machine_learning.hyperparameter_optimization"

# (child_module, parent_module) pairs representing the export chain.
# For each pair, every name in child.__all__ must appear in parent.__all__.
_INIT_CHAIN_PAIRS = [
    # Tabular classification
    (f"{_HP}.tabular.classification.test_functions", f"{_HP}.tabular.classification"),
    (f"{_HP}.tabular.classification", f"{_HP}.tabular"),
    # Tabular regression
    (f"{_HP}.tabular.regression.test_functions", f"{_HP}.tabular.regression"),
    (f"{_HP}.tabular.regression", f"{_HP}.tabular"),
    # Tabular -> hyperparameter_optimization
    (f"{_HP}.tabular", f"{_HP}"),
    # Image chain
    (f"{_HP}.image.classification.test_functions", f"{_HP}.image.classification"),
    (f"{_HP}.image.classification", f"{_HP}.image"),
    (f"{_HP}.image", f"{_HP}"),
    # Timeseries chain
    (f"{_HP}.timeseries", f"{_HP}"),
    # hyperparameter_optimization -> machine_learning
    (f"{_HP}", "surfaces.test_functions.machine_learning"),
]


def _short_name(module_path: str) -> str:
    """Shorten a module path for readable test IDs."""
    return module_path.replace(
        "surfaces.test_functions.machine_learning.hyperparameter_optimization",
        "hp_opt",
    ).replace(
        "surfaces.test_functions.machine_learning",
        "ml",
    )


@pytest.mark.ml
@pytest.mark.static
class TestMLInitChainConsistency:
    """Exports at each __init__.py level must propagate to the parent level."""

    @pytest.mark.parametrize(
        "child_path,parent_path",
        _INIT_CHAIN_PAIRS,
        ids=[f"{_short_name(c)} -> {_short_name(p)}" for c, p in _INIT_CHAIN_PAIRS],
    )
    def test_child_exports_in_parent(self, child_path, parent_path):
        """All names in child.__all__ must appear in parent.__all__.

        If a class is exported at a lower level but not at a higher level,
        users cannot import it from the standard entry point.
        """
        try:
            child = importlib.import_module(child_path)
            parent = importlib.import_module(parent_path)
        except ImportError as e:
            pytest.skip(f"Cannot import module: {e}")

        child_all = set(getattr(child, "__all__", []))
        parent_all = set(getattr(parent, "__all__", []))

        if not child_all:
            pytest.skip(f"{child_path} has no __all__")

        missing = child_all - parent_all
        if missing:
            pytest.fail(
                f"Exported in {_short_name(child_path)} but NOT in "
                f"{_short_name(parent_path)}:\n"
                f"  {sorted(missing)}\n"
                f"Add these to {parent_path.replace('.', '/')}/__init__.py"
            )
