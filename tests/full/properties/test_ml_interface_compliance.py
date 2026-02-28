"""ML-specific interface compliance tests.

Validates that every concrete ML function class follows the required
interface contract for the MachineLearningFunction hierarchy.

All tests are STATIC -- they inspect class definitions without
creating instances. This means they run fast and don't trigger
expensive model training or data loading.

What is checked
---------------
| Test Class               | What it enforces                              |
|--------------------------|-----------------------------------------------|
| TestMLRequiredAttributes | para_names, *_default, name/_name_/__name__   |
| TestMLRequiredMethods    | _ml_objective, _get_surrogate_params              |
| TestMLSearchSpaceConsistency | para_names <-> *_default cross-validation |
| TestMLUniqueIdentifiers  | _name_ uniqueness across all ML functions     |
"""

import inspect

import pytest

from surfaces.test_functions.machine_learning._base_machine_learning import (
    MachineLearningFunction,
)
from tests.full.properties._ml_discovery import ML_CLASSES, ml_class_id


def _has_dataset_param(cls):
    """Check if __init__ accepts a 'dataset' parameter (HP-opt pattern)."""
    try:
        sig = inspect.signature(cls.__init__)
        return "dataset" in sig.parameters
    except (ValueError, TypeError):
        return False


# Classes with known *_default violations (pre-existing, tracked for cleanup)
_KNOWN_DEFAULT_VIOLATIONS = {
    "FeatureEngineeringPipelineFunction",
    "MutualInfoFeatureSelectionFunction",
}

# Classes with dataset/cv that are missing _get_surrogate_params override
# (pre-existing, tracked for cleanup). New classes MUST NOT be added here.
_KNOWN_SURROGATE_VIOLATIONS = {
    "ClassificationPipelineFunction",
    "FeatureEngineeringPipelineFunction",
    "FeatureScalingPipelineFunction",
    "MutualInfoFeatureSelectionFunction",
    "PolynomialFeatureTransformationFunction",
    "RegressionPipelineFunction",
    "StackingEnsembleFunction",
    "VotingEnsembleFunction",
    "WeightedAveragingFunction",
}


@pytest.mark.ml
@pytest.mark.static
@pytest.mark.parametrize("cls", ML_CLASSES, ids=ml_class_id)
class TestMLRequiredAttributes:
    """Every ML function must have these class-level attributes."""

    def test_has_para_names(self, cls):
        """para_names must be a non-empty list of parameter names."""
        assert hasattr(cls, "para_names"), f"{cls.__name__}: Missing 'para_names' class attribute"
        assert isinstance(cls.para_names, list), (
            f"{cls.__name__}: 'para_names' must be a list, " f"got {type(cls.para_names).__name__}"
        )
        assert len(cls.para_names) > 0, f"{cls.__name__}: 'para_names' must not be empty"

    def test_has_default_for_each_param(self, cls):
        """Each entry in para_names must have a corresponding {param}_default list."""
        if cls.__name__ in _KNOWN_DEFAULT_VIOLATIONS:
            pytest.xfail(
                f"{cls.__name__}: Known missing *_default attributes (tracked for cleanup)"
            )

        for param in cls.para_names:
            attr = f"{param}_default"
            assert hasattr(cls, attr), (
                f"{cls.__name__}: Missing '{attr}' for param '{param}' " f"declared in para_names"
            )
            val = getattr(cls, attr)
            assert isinstance(val, list), (
                f"{cls.__name__}: '{attr}' must be a list, " f"got {type(val).__name__}"
            )
            assert len(val) > 0, f"{cls.__name__}: '{attr}' must not be empty"

    def test_has_name_triple(self, cls):
        """Must have name (human-readable), _name_ (internal ID), __name__ (class name)."""
        assert hasattr(cls, "name") and isinstance(
            cls.name, str
        ), f"{cls.__name__}: Missing or non-string 'name' attribute"
        assert hasattr(cls, "_name_") and isinstance(cls._name_, str), (
            f"{cls.__name__}: Missing or non-string '_name_' attribute "
            f"(used for surrogate model loading)"
        )
        assert hasattr(cls, "__name__") and isinstance(
            cls.__name__, str
        ), f"{cls.__name__}: Missing or non-string '__name__' attribute"


@pytest.mark.ml
@pytest.mark.static
@pytest.mark.parametrize("cls", ML_CLASSES, ids=ml_class_id)
class TestMLRequiredMethods:
    """Every ML function must implement these methods."""

    def test_implements_ml_objective(self, cls):
        """_ml_objective must be implemented, not just raise."""
        assert hasattr(cls, "_ml_objective"), f"{cls.__name__}: Missing '_ml_objective' method"
        try:
            source = inspect.getsource(cls._ml_objective)
            assert "NotImplementedError" not in source, (
                f"{cls.__name__}: _ml_objective still raises "
                f"NotImplementedError instead of providing an implementation"
            )
        except (TypeError, OSError):
            pass  # Cannot get source, skip source-level check

    def test_overrides_get_surrogate_params(self, cls):
        """_get_surrogate_params must be overridden for HP-opt functions.

        Applies to functions with a 'dataset' parameter in __init__ (the
        hyperparameter optimization pattern). These classes store fixed
        parameters (dataset, cv, n_components) outside the search space,
        and the surrogate model needs them for correct predictions.

        The base MachineLearningFunction._get_surrogate_params only passes
        through search parameters. Without the override, surrogate
        predictions silently produce wrong results.
        """
        if not _has_dataset_param(cls):
            pytest.skip("Not an HP-opt function (no 'dataset' parameter)")

        if cls.__name__ in _KNOWN_SURROGATE_VIOLATIONS:
            pytest.xfail(
                f"{cls.__name__}: Known missing _get_surrogate_params (tracked for cleanup)"
            )

        has_override = False
        for klass in cls.__mro__:
            if klass is MachineLearningFunction:
                break
            if "_get_surrogate_params" in klass.__dict__:
                has_override = True
                break

        assert has_override, (
            f"{cls.__name__}: Must override '_get_surrogate_params' to include "
            f"fixed parameters (dataset, cv, etc.) for surrogate model support. "
            f"Example: return {{**params, 'dataset': self.dataset, 'cv': self.cv}}"
        )


@pytest.mark.ml
@pytest.mark.static
@pytest.mark.parametrize("cls", ML_CLASSES, ids=ml_class_id)
class TestMLSearchSpaceConsistency:
    """Cross-validate para_names against *_default attributes."""

    def test_no_orphan_defaults(self, cls):
        """No *_default attribute should exist without a matching para_names entry.

        Catches the case where a developer adds a default list but forgets
        to include the parameter in para_names (or removes from para_names
        but leaves the default).
        """
        for attr_name in vars(cls):
            if attr_name.endswith("_default") and not attr_name.startswith("_"):
                param_name = attr_name[:-8]  # strip "_default"
                value = getattr(cls, attr_name)
                if param_name and isinstance(value, list):
                    assert param_name in cls.para_names, (
                        f"{cls.__name__}: Has '{attr_name}' attribute but "
                        f"'{param_name}' is not in para_names = {cls.para_names}"
                    )


@pytest.mark.ml
@pytest.mark.static
class TestMLUniqueIdentifiers:
    """Identifiers must be unique across all ML functions."""

    def test_unique_name_identifiers(self):
        """Each ML function must have a unique _name_ (used for surrogate loading)."""
        seen = {}
        duplicates = []
        for cls in ML_CLASSES:
            name = cls._name_
            if name in seen:
                duplicates.append(
                    f"  '{name}' used by both " f"{seen[name].__name__} and {cls.__name__}"
                )
            seen[name] = cls

        assert not duplicates, (
            "_name_ identifiers must be unique "
            "(used for surrogate model file lookup):\n" + "\n".join(duplicates)
        )
