"""Tests for SpecAccessor with ML functions (requires sklearn)."""

from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction


class TestSpecWithMLFunctions:
    """Test SpecAccessor behavior with machine learning functions."""

    def test_ml_spec_has_expected_keys(self):
        """ML function spec contains ML-specific keys."""
        func = KNeighborsClassifierFunction()
        spec = func.spec.as_dict()
        assert isinstance(spec, dict)
        # ML functions have their own spec values
        assert func.spec.scalable is False

    def test_ml_spec_dict_protocol(self):
        """Dict-like access works on ML function specs."""
        func = KNeighborsClassifierFunction()
        assert "continuous" in func.spec
        assert func.spec.get("nonexistent", "default") == "default"

    def test_ml_f_global_via_spec(self):
        """f_global is accessible via spec accessor for ML functions."""
        func = KNeighborsClassifierFunction()
        # ML functions may or may not have f_global set
        _ = func.spec.f_global  # Should not raise
