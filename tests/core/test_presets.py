# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for presets module."""

import pytest


class TestPresetImports:
    """Test that all presets can be imported."""

    def test_import_presets_module(self):
        from surfaces import presets

        assert presets is not None

    def test_import_quick(self):
        from surfaces.presets import quick

        assert isinstance(quick, list)
        assert len(quick) == 5

    def test_import_standard(self):
        from surfaces.presets import standard

        assert isinstance(standard, list)
        assert len(standard) == 15

    def test_import_algebraic_2d(self):
        from surfaces.presets import algebraic_2d

        assert isinstance(algebraic_2d, list)
        assert len(algebraic_2d) == 18

    def test_import_algebraic_nd(self):
        from surfaces.presets import algebraic_nd

        assert isinstance(algebraic_nd, list)
        assert len(algebraic_nd) == 5

    def test_import_bbob(self):
        from surfaces.presets import bbob

        assert isinstance(bbob, list)
        assert len(bbob) == 24

    def test_import_cec2014(self):
        from surfaces.presets import cec2014

        assert isinstance(cec2014, list)
        assert len(cec2014) == 30

    def test_import_cec2017(self):
        from surfaces.presets import cec2017

        assert isinstance(cec2017, list)
        assert len(cec2017) == 10

    def test_import_engineering(self):
        from surfaces.presets import engineering

        assert isinstance(engineering, list)
        assert len(engineering) == 5


class TestPresetContents:
    """Test that preset contents are valid function classes."""

    def test_quick_contains_classes(self):
        from surfaces.presets import quick

        for func_class in quick:
            assert callable(func_class)
            assert hasattr(func_class, "__name__")

    def test_standard_contains_classes(self):
        from surfaces.presets import standard

        for func_class in standard:
            assert callable(func_class)
            assert hasattr(func_class, "__name__")

    def test_bbob_contains_classes(self):
        from surfaces.presets import bbob

        for func_class in bbob:
            assert callable(func_class)
            assert hasattr(func_class, "__name__")


class TestInstantiate:
    """Test the instantiate helper function."""

    def test_instantiate_quick(self):
        from surfaces.presets import instantiate, quick

        functions = instantiate(quick, n_dim=5)
        assert len(functions) == 5
        for func in functions:
            assert hasattr(func, "search_space")
            assert hasattr(func, "__call__")

    def test_instantiate_quick_evaluation(self):
        from surfaces.presets import instantiate, quick

        functions = instantiate(quick, n_dim=5)
        for func in functions:
            params = {
                name: (bounds[0] + bounds[1]) / 2 for name, bounds in func.search_space.items()
            }
            result = func(params)
            assert isinstance(result, (int, float))

    def test_instantiate_algebraic_2d(self):
        """2D functions should have 2 dimensions regardless of n_dim param."""
        from surfaces.presets import algebraic_2d, instantiate

        functions = instantiate(algebraic_2d[:3])
        for func in functions:
            assert len(func.search_space) == 2

    def test_instantiate_algebraic_nd(self):
        """ND functions should have the specified n_dim."""
        from surfaces.presets import algebraic_nd, instantiate

        for n_dim in [5, 10, 20]:
            functions = instantiate(algebraic_nd, n_dim=n_dim)
            for func in functions:
                assert len(func.search_space) == n_dim

    def test_instantiate_bbob(self):
        from surfaces.presets import bbob, instantiate

        functions = instantiate(bbob[:3], n_dim=10)
        for func in functions:
            assert len(func.search_space) == 10

    def test_instantiate_standard_mixed(self):
        """Standard preset has mixed 2D and ND functions."""
        from surfaces.presets import instantiate, standard

        functions = instantiate(standard, n_dim=5)
        dims = [len(func.search_space) for func in functions]
        assert 2 in dims  # Has 2D functions
        assert 5 in dims  # Has 5D functions


class TestDirectInstantiation:
    """Test direct instantiation patterns."""

    def test_bbob_direct_instantiation(self):
        from surfaces.presets import bbob

        for FuncClass in bbob[:3]:
            func = FuncClass(n_dim=5, instance=1)
            assert hasattr(func, "search_space")
            assert len(func.search_space) == 5

    def test_algebraic_nd_direct_instantiation(self):
        from surfaces.presets import algebraic_nd

        for FuncClass in algebraic_nd:
            for n_dim in [2, 5, 10]:
                func = FuncClass(n_dim=n_dim)
                assert len(func.search_space) == n_dim

    def test_algebraic_2d_direct_instantiation(self):
        from surfaces.presets import algebraic_2d

        for FuncClass in algebraic_2d[:3]:
            func = FuncClass()
            assert len(func.search_space) == 2


class TestUtilityFunctions:
    """Test utility functions."""

    def test_get_valid(self):
        from surfaces.presets import get

        preset = get("quick")
        assert len(preset) == 5

        preset = get("standard")
        assert len(preset) == 15

    def test_get_invalid(self):
        from surfaces.presets import get

        with pytest.raises(ValueError, match="Unknown preset"):
            get("nonexistent")

    def test_list_presets(self):
        from surfaces.presets import list_presets

        presets = list_presets()
        assert isinstance(presets, dict)
        assert "quick" in presets
        assert "standard" in presets
        assert presets["quick"] == 5
        assert presets["bbob"] == 24


class TestNoDuplicates:
    """Test that presets don't have duplicate entries."""

    def test_quick_no_duplicates(self):
        from surfaces.presets import quick

        assert len(quick) == len(set(quick))

    def test_standard_no_duplicates(self):
        from surfaces.presets import standard

        assert len(standard) == len(set(standard))

    def test_bbob_no_duplicates(self):
        from surfaces.presets import bbob

        assert len(bbob) == len(set(bbob))

    def test_cec2014_no_duplicates(self):
        from surfaces.presets import cec2014

        assert len(cec2014) == len(set(cec2014))


class TestEngineeringPreset:
    """Test engineering functions preset specifics."""

    def test_engineering_instantiate(self):
        from surfaces.presets import engineering, instantiate

        functions = instantiate(engineering)
        assert len(functions) == 5

    def test_engineering_have_constraints(self):
        from surfaces.presets import engineering, instantiate

        functions = instantiate(engineering)
        for func in functions:
            assert hasattr(func, "constraints")
            assert hasattr(func, "is_feasible")
            assert hasattr(func, "constraint_violations")
