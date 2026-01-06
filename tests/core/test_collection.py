# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for collection module."""

import pytest


class TestCollectionImports:
    """Test that collection can be imported."""

    def test_import_collection(self):
        from surfaces import collection

        assert collection is not None

    def test_collection_is_singleton(self):
        from surfaces import collection
        from surfaces.collection import _CollectionSingleton

        assert isinstance(collection, _CollectionSingleton)


class TestPredefinedCollections:
    """Test predefined collections."""

    def test_quick(self):
        from surfaces import collection

        assert len(collection.quick) == 5
        assert collection.quick._name == "quick"

    def test_standard(self):
        from surfaces import collection

        assert len(collection.standard) == 15
        assert collection.standard._name == "standard"

    def test_bbob(self):
        from surfaces import collection

        assert len(collection.bbob) == 24
        assert collection.bbob._name == "bbob"

    def test_cec2014(self):
        from surfaces import collection

        assert len(collection.cec2014) == 30
        assert collection.cec2014._name == "cec2014"

    def test_cec2017(self):
        from surfaces import collection

        assert len(collection.cec2017) == 10
        assert collection.cec2017._name == "cec2017"

    def test_engineering(self):
        from surfaces import collection

        assert len(collection.engineering) == 5
        assert collection.engineering._name == "engineering"

    def test_algebraic_2d(self):
        from surfaces import collection

        assert len(collection.algebraic_2d) == 18
        assert collection.algebraic_2d._name == "algebraic_2d"

    def test_algebraic_nd(self):
        from surfaces import collection

        assert len(collection.algebraic_nd) == 5
        assert collection.algebraic_nd._name == "algebraic_nd"


class TestCollectionSingleton:
    """Test that collection singleton contains all functions."""

    def test_collection_is_iterable(self):
        from surfaces import collection

        count = 0
        for _ in collection:
            count += 1
        assert count > 50

    def test_collection_has_len(self):
        from surfaces import collection

        assert len(collection) > 50

    def test_collection_contains_algebraic(self):
        from surfaces import collection

        names = collection.names
        assert "SphereFunction" in names
        assert "RastriginFunction" in names

    def test_collection_contains_bbob(self):
        from surfaces import collection

        names = collection.names
        assert "Sphere" in names  # BBOB Sphere

    def test_collection_contains_engineering(self):
        from surfaces import collection

        names = collection.names
        assert "WeldedBeamFunction" in names


class TestCollectionFilter:
    """Test collection.filter() method."""

    def test_filter_by_unimodal(self):
        from surfaces import collection

        result = collection.filter(unimodal=True)
        assert len(result) > 0
        assert "SphereFunction" in result.names

    def test_filter_by_convex(self):
        from surfaces import collection

        result = collection.filter(convex=True)
        assert len(result) > 0
        assert "SphereFunction" in result.names

    def test_filter_by_separable(self):
        from surfaces import collection

        result = collection.filter(separable=True)
        assert len(result) > 0

    def test_filter_by_scalable(self):
        from surfaces import collection

        result = collection.filter(scalable=True)
        assert len(result) > 0

    def test_filter_by_category_algebraic(self):
        from surfaces import collection

        result = collection.filter(category="algebraic")
        assert len(result) == 28  # All algebraic functions
        assert all("algebraic" in f.__module__ for f in result)

    def test_filter_by_category_bbob(self):
        from surfaces import collection

        result = collection.filter(category="bbob")
        assert len(result) == 24
        assert all("bbob" in f.__module__ for f in result)

    def test_filter_by_category_engineering(self):
        from surfaces import collection

        result = collection.filter(category="engineering")
        assert len(result) == 5
        assert all("engineering" in f.__module__ for f in result)

    def test_filter_multiple_criteria(self):
        from surfaces import collection

        result = collection.filter(unimodal=True, convex=True)
        assert len(result) > 0
        # All results should be both unimodal and convex
        assert "SphereFunction" in result.names

    def test_filter_returns_collection(self):
        from surfaces import collection

        result = collection.filter(unimodal=True)
        # Result should be a collection-like object
        assert hasattr(result, "filter")
        assert hasattr(result, "search")
        assert hasattr(result, "describe")


class TestCollectionSearch:
    """Test collection.search() method."""

    def test_search_by_name(self):
        from surfaces import collection

        result = collection.search("sphere")
        assert len(result) >= 1
        assert any("Sphere" in name for name in result.names)

    def test_search_case_insensitive(self):
        from surfaces import collection

        result1 = collection.search("SPHERE")
        result2 = collection.search("sphere")
        result3 = collection.search("Sphere")
        assert result1.names == result2.names == result3.names

    def test_search_partial_match(self):
        from surfaces import collection

        result = collection.search("rastrigin")
        assert len(result) >= 2  # Multiple Rastrigin variants

    def test_search_no_match(self):
        from surfaces import collection

        result = collection.search("nonexistent_xyz_123")
        assert len(result) == 0

    def test_search_returns_collection(self):
        from surfaces import collection

        result = collection.search("sphere")
        # Result should be a collection-like object
        assert hasattr(result, "filter")
        assert hasattr(result, "search")
        assert hasattr(result, "describe")


class TestCollectionChaining:
    """Test method chaining on collections."""

    def test_filter_chain(self):
        from surfaces import collection

        result = collection.filter(category="algebraic").filter(unimodal=True)
        assert len(result) > 0
        # All should be algebraic and unimodal
        for f in result:
            assert "algebraic" in f.__module__

    def test_search_then_filter(self):
        from surfaces import collection

        result = collection.search("rastrigin").filter(category="algebraic")
        assert len(result) >= 1
        assert "RastriginFunction" in result.names


class TestCollectionIteration:
    """Test iteration over collections."""

    def test_iteration_predefined(self):
        from surfaces import collection

        count = 0
        for func_cls in collection.quick:
            count += 1
            assert callable(func_cls)
        assert count == 5

    def test_iteration_singleton(self):
        from surfaces import collection

        count = 0
        for func_cls in collection:
            count += 1
            assert callable(func_cls)
        assert count > 50

    def test_len(self):
        from surfaces import collection

        assert len(collection.quick) == 5
        assert len(collection.bbob) == 24
        assert len(collection) > 50

    def test_contains(self):
        from surfaces.test_functions import SphereFunction

        from surfaces import collection

        assert SphereFunction in collection.quick
        assert SphereFunction in collection

    def test_indexing(self):
        from surfaces import collection

        first = collection.quick[0]
        assert callable(first)
        last = collection.quick[-1]
        assert callable(last)
        # Also test singleton indexing
        first_all = collection[0]
        assert callable(first_all)


class TestCollectionSetOperations:
    """Test set operations on collections."""

    def test_union(self):
        from surfaces import collection

        combined = collection.quick + collection.engineering
        assert len(combined) == 10  # 5 + 5, no overlap

    def test_union_no_duplicates(self):
        from surfaces import collection

        combined = collection.quick + collection.quick
        assert len(combined) == 5  # No duplicates

    def test_intersection(self):
        from surfaces import collection

        result = collection.quick & collection.filter(unimodal=True)
        # SphereFunction and RosenbrockFunction are unimodal in quick
        assert len(result) >= 2

    def test_difference(self):
        from surfaces import collection

        result = collection.quick - collection.filter(unimodal=True)
        # Functions in quick that are NOT unimodal
        for f in result:
            assert f.__name__ not in ["SphereFunction", "RosenbrockFunction"]


class TestCollectionProperties:
    """Test collection properties and methods."""

    def test_names_property(self):
        from surfaces import collection

        names = collection.quick.names
        assert isinstance(names, list)
        assert len(names) == 5
        assert "SphereFunction" in names

    def test_describe(self):
        from surfaces import collection

        desc = collection.quick.describe()
        assert isinstance(desc, str)
        assert "SphereFunction" in desc
        assert "Collection: 5 functions" in desc

    def test_repr(self):
        from surfaces import collection

        repr_str = repr(collection.quick)
        assert "Collection" in repr_str
        assert "5 functions" in repr_str
        assert "quick" in repr_str


class TestModuleFunctions:
    """Test module-level functions."""

    def test_categories(self):
        from surfaces import collection

        cats = collection.categories()
        assert isinstance(cats, list)
        assert "algebraic" in cats
        assert "bbob" in cats
        assert "engineering" in cats

    def test_properties(self):
        from surfaces import collection

        props = collection.properties()
        assert isinstance(props, list)
        assert "unimodal" in props
        assert "convex" in props
        assert "category" in props
        assert "n_dim" in props


class TestCollectionInstantiation:
    """Test that collection functions can be instantiated."""

    def test_instantiate_quick(self):
        from surfaces import collection

        for func_cls in collection.quick:
            # Try to instantiate (some need n_dim)
            try:
                func = func_cls(n_dim=5)
            except TypeError:
                func = func_cls()
            assert hasattr(func, "search_space")
            assert callable(func)

    def test_instantiate_and_evaluate(self):
        from surfaces import collection

        for func_cls in collection.quick:
            try:
                func = func_cls(n_dim=5)
            except TypeError:
                func = func_cls()

            # Evaluate at midpoint
            params = {name: (b[0] + b[1]) / 2 for name, b in func.search_space.items()}
            result = func(params)
            assert isinstance(result, (int, float))

    def test_instantiate_algebraic_2d(self):
        """2D functions should have 2 dimensions."""
        from surfaces import collection

        for func_cls in collection.algebraic_2d:
            func = func_cls()
            assert len(func.search_space) == 2

    def test_instantiate_algebraic_nd(self):
        """ND functions should have the specified n_dim."""
        from surfaces import collection

        for n_dim in [5, 10, 20]:
            for func_cls in collection.algebraic_nd:
                func = func_cls(n_dim=n_dim)
                assert len(func.search_space) == n_dim


class TestShow:
    """Test show method."""

    def test_show(self):
        from surfaces import collection

        suites = collection.show()
        assert isinstance(suites, dict)
        assert "quick" in suites
        assert "standard" in suites
        assert suites["quick"] == 5
        assert suites["bbob"] == 24

    def test_show_all_present(self):
        from surfaces import collection

        suites = collection.show()
        expected = [
            "quick",
            "standard",
            "algebraic_2d",
            "algebraic_nd",
            "bbob",
            "cec2014",
            "cec2017",
            "engineering",
        ]
        for name in expected:
            assert name in suites


class TestNoDuplicates:
    """Test that collections don't have duplicate entries."""

    def test_quick_no_duplicates(self):
        from surfaces import collection

        funcs = list(collection.quick)
        assert len(funcs) == len(set(funcs))

    def test_singleton_no_duplicates(self):
        from surfaces import collection

        funcs = list(collection)
        assert len(funcs) == len(set(funcs))

    def test_bbob_no_duplicates(self):
        from surfaces import collection

        funcs = list(collection.bbob)
        assert len(funcs) == len(set(funcs))
