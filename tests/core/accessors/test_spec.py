"""Tests for SpecAccessor: dict-like access to function characteristics."""

import pytest

from surfaces.test_functions._accessors._spec import SpecAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestSpecType:
    """Test that spec returns the correct accessor type."""

    def test_spec_returns_accessor(self):
        """func.spec returns a SpecAccessor, not a plain dict."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.spec, SpecAccessor)

    def test_as_dict_returns_plain_dict(self):
        """func.spec.as_dict() returns a plain dict."""
        func = SphereFunction(n_dim=2)
        result = func.spec.as_dict()
        assert isinstance(result, dict)
        assert not isinstance(result, SpecAccessor)

    def test_spec_is_cached(self):
        """Repeated access returns the same accessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.spec is func.spec


class TestSpecProtocol:
    """Test dict-like protocol on SpecAccessor."""

    def test_getitem(self):
        """Bracket access retrieves spec values."""
        func = SphereFunction(n_dim=2)
        assert func.spec["convex"] is True

    def test_getitem_nonexistent_raises_key_error(self):
        """Accessing a nonexistent key raises KeyError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(KeyError):
            func.spec["nonexistent_key"]

    def test_contains(self):
        """'in' operator checks for key presence."""
        func = SphereFunction(n_dim=2)
        assert "convex" in func.spec
        assert "nonexistent_key" not in func.spec

    def test_get_with_default(self):
        """get() returns default when key is missing."""
        func = SphereFunction(n_dim=2)
        assert func.spec.get("convex") is True
        assert func.spec.get("nonexistent_key", "fallback") == "fallback"

    def test_get_without_default_returns_none(self):
        """get() returns None when key is missing and no default given."""
        func = SphereFunction(n_dim=2)
        assert func.spec.get("nonexistent_key") is None


class TestSpecProperties:
    """Test typed property accessors on SpecAccessor."""

    def test_convex(self):
        """SphereFunction is convex."""
        func = SphereFunction(n_dim=2)
        assert func.spec.convex is True

    def test_unimodal(self):
        """SphereFunction is unimodal."""
        func = SphereFunction(n_dim=2)
        assert func.spec.unimodal is True

    def test_separable(self):
        """SphereFunction is separable."""
        func = SphereFunction(n_dim=2)
        assert func.spec.separable is True

    def test_continuous(self):
        """SphereFunction is continuous (inherited from AlgebraicFunction)."""
        func = SphereFunction(n_dim=2)
        assert func.spec.continuous is True

    def test_differentiable(self):
        """SphereFunction is differentiable (inherited from AlgebraicFunction)."""
        func = SphereFunction(n_dim=2)
        assert func.spec.differentiable is True

    def test_scalable(self):
        """SphereFunction is scalable."""
        func = SphereFunction(n_dim=2)
        assert func.spec.scalable is True

    def test_n_objectives(self):
        """SphereFunction has 1 objective (inherited from BaseTestFunction)."""
        func = SphereFunction(n_dim=2)
        assert func.spec.n_objectives == 1

    def test_default_bounds(self):
        """SphereFunction has default_bounds (-5.0, 5.0)."""
        func = SphereFunction(n_dim=2)
        assert func.spec.default_bounds == (-5.0, 5.0)


class TestSpecGlobalOptimum:
    """Test global optimum access through SpecAccessor."""

    def test_f_global(self):
        """SphereFunction has f_global=0.0."""
        func = SphereFunction(n_dim=2)
        assert func.spec.f_global == 0.0

    def test_x_global(self):
        """SphereFunction has x_global at the origin."""
        func = SphereFunction(n_dim=3)
        assert func.spec.x_global == (0.0, 0.0, 0.0)


class TestSpecMROmerging:
    """Test that _spec dicts merge correctly through the MRO.

    SphereFunction overrides convex, unimodal, separable, scalable.
    AlgebraicFunction defines default_bounds, continuous, differentiable.
    BaseTestFunction defines n_dim, n_objectives, func_id, and boolean defaults.
    """

    def test_child_overrides_parent(self):
        """SphereFunction._spec overrides BaseTestFunction._spec defaults."""
        func = SphereFunction(n_dim=2)
        spec = func.spec.as_dict()

        # SphereFunction sets these to True; BaseTestFunction defaults are False
        assert spec["convex"] is True
        assert spec["unimodal"] is True
        assert spec["separable"] is True
        assert spec["scalable"] is True

    def test_parent_keys_preserved(self):
        """Keys only defined in parent classes are still accessible."""
        func = SphereFunction(n_dim=2)
        spec = func.spec.as_dict()

        # These come from BaseTestFunction._spec
        assert "n_objectives" in spec
        assert "func_id" in spec

    def test_intermediate_class_keys_preserved(self):
        """Keys from AlgebraicFunction (middle of MRO) are present."""
        func = SphereFunction(n_dim=2)
        spec = func.spec.as_dict()
        assert spec["continuous"] is True
        assert spec["differentiable"] is True
