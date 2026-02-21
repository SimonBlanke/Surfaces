"""Tests for MetaAccessor: read-only proxy to class-level metadata."""

import pytest

from surfaces.test_functions._accessors._meta import MetaAccessor
from surfaces.test_functions.algebraic import SphereFunction


class TestMetaProperties:
    """Test MetaAccessor typed property access."""

    def test_name(self):
        """func.meta.name returns the function name."""
        func = SphereFunction(n_dim=2)
        assert func.meta.name == "Sphere Function"

    def test_latex_formula(self):
        """func.meta.latex_formula returns the LaTeX string."""
        func = SphereFunction(n_dim=2)
        assert func.meta.latex_formula is not None
        assert r"\sum" in func.meta.latex_formula

    def test_reference(self):
        """func.meta.reference returns the reference or None."""
        func = SphereFunction(n_dim=2)
        # SphereFunction.reference is None
        assert func.meta.reference is None

    def test_reference_url(self):
        """func.meta.reference_url returns the URL string."""
        func = SphereFunction(n_dim=2)
        assert func.meta.reference_url is not None
        assert "sfu.ca" in func.meta.reference_url

    def test_tagline(self):
        """func.meta.tagline returns the tagline string."""
        func = SphereFunction(n_dim=2)
        assert func.meta.tagline is not None
        assert isinstance(func.meta.tagline, str)

    def test_func_id(self):
        """func.meta.func_id returns None for SphereFunction."""
        func = SphereFunction(n_dim=2)
        # SphereFunction does not have a func_id class attribute
        assert func.meta.func_id is None


class TestMetaReadOnly:
    """Test that MetaAccessor properties are read-only."""

    def test_name_not_settable(self):
        """Setting meta.name raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.name = "other"

    def test_latex_formula_not_settable(self):
        """Setting meta.latex_formula raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.latex_formula = "other"

    def test_reference_not_settable(self):
        """Setting meta.reference raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.reference = "other"

    def test_reference_url_not_settable(self):
        """Setting meta.reference_url raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.reference_url = "other"

    def test_tagline_not_settable(self):
        """Setting meta.tagline raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.tagline = "other"

    def test_func_id_not_settable(self):
        """Setting meta.func_id raises AttributeError."""
        func = SphereFunction(n_dim=2)
        with pytest.raises(AttributeError):
            func.meta.func_id = "other"


class TestMetaCaching:
    """Test accessor caching on the function instance."""

    def test_accessor_is_cached(self):
        """Repeated access returns the same MetaAccessor instance."""
        func = SphereFunction(n_dim=2)
        assert func.meta is func.meta

    def test_accessor_type(self):
        """func.meta is a MetaAccessor."""
        func = SphereFunction(n_dim=2)
        assert isinstance(func.meta, MetaAccessor)
