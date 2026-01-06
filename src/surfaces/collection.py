# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Function collection for discovering and filtering test functions.

This module provides a unified interface to browse, filter, and select
test functions from the Surfaces library.

Examples
--------
>>> from surfaces import collection
>>>
>>> # collection is iterable and contains all functions
>>> len(collection)                       # 118 functions
>>> for func_cls in collection:
...     print(func_cls.__name__)
>>>
>>> # Filter and search
>>> collection.filter(unimodal=True)      # Filter by properties
>>> collection.filter(category="algebraic")
>>> collection.search("rastrigin")        # Search in names/taglines
>>>
>>> # Predefined collections
>>> collection.quick                      # 5 functions for smoke tests
>>> collection.standard                   # 15 functions for academic comparison
>>> collection.bbob                       # 24 COCO/BBOB functions
>>> collection.engineering                # 5 constrained problems
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type

if TYPE_CHECKING:
    from .test_functions._base_test_function import BaseTestFunction


class Collection:
    """A filterable collection of test functions.

    This class represents a set of test function classes that can be
    iterated, filtered, and combined using set operations.

    Parameters
    ----------
    functions : list of type
        List of test function classes.
    name : str, optional
        Name of this collection (for display purposes).

    Examples
    --------
    >>> from surfaces import collection
    >>> unimodal = collection.filter(unimodal=True)
    >>> len(unimodal)
    >>> for func_cls in unimodal:
    ...     print(func_cls.__name__)
    """

    def __init__(
        self,
        functions: List[Type["BaseTestFunction"]],
        name: Optional[str] = None,
    ) -> None:
        self._functions = list(functions)
        self._name = name

    def __iter__(self) -> Iterator[Type["BaseTestFunction"]]:
        return iter(self._functions)

    def __len__(self) -> int:
        return len(self._functions)

    def __repr__(self) -> str:
        name_part = f"'{self._name}'" if self._name else ""
        return f"Collection({name_part}, {len(self)} functions)"

    def __contains__(self, item: Type["BaseTestFunction"]) -> bool:
        return item in self._functions

    def __getitem__(self, index: int) -> Type["BaseTestFunction"]:
        return self._functions[index]

    def __add__(self, other: "Collection") -> "Collection":
        """Union of two collections."""
        combined = list(dict.fromkeys(self._functions + other._functions))
        return Collection(combined)

    def __and__(self, other: "Collection") -> "Collection":
        """Intersection of two collections."""
        other_set = set(other._functions)
        intersection = [f for f in self._functions if f in other_set]
        return Collection(intersection)

    def __sub__(self, other: "Collection") -> "Collection":
        """Difference of two collections."""
        other_set = set(other._functions)
        difference = [f for f in self._functions if f not in other_set]
        return Collection(difference)

    @property
    def names(self) -> List[str]:
        """List of function class names."""
        return [f.__name__ for f in self._functions]

    def filter(self, **criteria: Any) -> "Collection":
        """Filter functions by properties.

        Parameters
        ----------
        **criteria : keyword arguments
            Filter criteria. Supported filters:
            - category: str - "algebraic", "bbob", "cec", "engineering", "ml"
            - n_dim: int - Number of dimensions (exact match or None for variable)
            - unimodal: bool - Single global optimum
            - convex: bool - Convex function
            - separable: bool - Variables are independent
            - scalable: bool - Can be used with any n_dim
            - continuous: bool - Continuous function
            - differentiable: bool - Differentiable function

        Returns
        -------
        Collection
            New collection with filtered functions.

        Examples
        --------
        >>> collection.filter(unimodal=True)
        >>> collection.filter(category="algebraic", convex=True)
        """
        result = self._functions

        for key, value in criteria.items():
            if key == "category":
                result = [f for f in result if _get_category(f) == value]
            elif key == "n_dim":
                result = [f for f in result if _get_n_dim(f) == value]
            else:
                # Filter by spec property
                result = [f for f in result if _get_spec_value(f, key) == value]

        return Collection(result)

    def search(self, query: str) -> "Collection":
        """Search functions by name or tagline.

        Parameters
        ----------
        query : str
            Search string (case-insensitive).

        Returns
        -------
        Collection
            New collection with matching functions.

        Examples
        --------
        >>> collection.search("rastrigin")
        >>> collection.search("multimodal")
        """
        query_lower = query.lower()
        result = []

        for func_cls in self._functions:
            # Search in class name
            if query_lower in func_cls.__name__.lower():
                result.append(func_cls)
                continue

            # Search in name attribute
            name = getattr(func_cls, "name", "")
            if name and query_lower in name.lower():
                result.append(func_cls)
                continue

            # Search in tagline
            tagline = getattr(func_cls, "tagline", "")
            if tagline and query_lower in tagline.lower():
                result.append(func_cls)
                continue

        return Collection(result)

    def describe(self) -> str:
        """Return a formatted description of all functions in the collection.

        Returns
        -------
        str
            Multi-line string with function details.
        """
        lines = [f"Collection: {len(self)} functions", "=" * 50]

        for func_cls in self._functions:
            name = func_cls.__name__
            category = _get_category(func_cls)
            n_dim = _get_n_dim(func_cls)
            dim_str = str(n_dim) if n_dim else "N"

            spec = _get_merged_spec(func_cls)
            props = []
            if spec.get("unimodal"):
                props.append("unimodal")
            if spec.get("convex"):
                props.append("convex")
            if spec.get("separable"):
                props.append("separable")
            if spec.get("scalable"):
                props.append("scalable")

            props_str = ", ".join(props) if props else "-"
            lines.append(f"{name:<35} {category:<12} {dim_str:>3}D  [{props_str}]")

        return "\n".join(lines)

    def categories(self) -> List[str]:
        """Get list of categories in this collection.

        Returns
        -------
        list of str
            Category names.
        """
        cats = set(_get_category(f) for f in self._functions)
        return sorted(cats)

    def properties(self) -> List[str]:
        """Get list of filterable properties.

        Returns
        -------
        list of str
            Property names that can be used with filter().
        """
        return [
            "category",
            "n_dim",
            "unimodal",
            "convex",
            "separable",
            "scalable",
            "continuous",
            "differentiable",
        ]


# =============================================================================
# Helper functions for introspection
# =============================================================================


def _get_merged_spec(func_cls: Type) -> Dict[str, Any]:
    """Get merged spec from class hierarchy."""
    result = {}
    for klass in reversed(func_cls.__mro__):
        if hasattr(klass, "_spec"):
            result.update(klass._spec)
    return result


def _get_spec_value(func_cls: Type, key: str) -> Any:
    """Get a specific spec value from a function class."""
    spec = _get_merged_spec(func_cls)
    return spec.get(key)


def _get_n_dim(func_cls: Type) -> Optional[int]:
    """Get n_dim from spec or class attribute."""
    # Check spec first
    spec = _get_merged_spec(func_cls)
    if "n_dim" in spec and spec["n_dim"] is not None:
        return spec["n_dim"]

    # Check class attribute (for fixed-dimension functions)
    if hasattr(func_cls, "n_dim") and func_cls.n_dim is not None:
        return func_cls.n_dim

    return None


def _get_category(func_cls: Type) -> str:
    """Determine the category of a function class."""
    module = func_cls.__module__

    if ".algebraic" in module:
        return "algebraic"
    elif ".bbob" in module:
        return "bbob"
    elif ".cec" in module:
        return "cec"
    elif ".engineering" in module:
        return "engineering"
    elif ".machine_learning" in module:
        return "ml"
    else:
        return "other"


# =============================================================================
# Singleton Collection with predefined subsets
# =============================================================================


class _CollectionSingleton(Collection):
    """Singleton collection containing all test functions.

    This class extends Collection with lazy-loaded predefined subsets
    accessible as properties (quick, standard, bbob, etc.).
    """

    def __init__(self) -> None:
        # Lazy initialization - don't build registry in __init__
        self._functions = None
        self._name = "all"
        self._predefined_cache: Dict[str, Collection] = {}

    def _ensure_initialized(self) -> None:
        """Lazily initialize the function registry."""
        if self._functions is not None:
            return

        functions = []
        seen = set()

        def add_functions(func_list: List[Type]) -> None:
            for f in func_list:
                if f not in seen:
                    seen.add(f)
                    functions.append(f)

        # Algebraic functions (always available)
        from .test_functions.algebraic import algebraic_functions

        add_functions(algebraic_functions)

        # BBOB functions (always available)
        from .test_functions.bbob import BBOB_FUNCTIONS

        add_functions(list(BBOB_FUNCTIONS.values()))

        # Engineering functions (always available)
        from .test_functions.engineering import engineering_functions

        add_functions(engineering_functions)

        # CEC functions from presets
        from .presets import suites as _suites

        add_functions(_suites.cec2014)
        add_functions(_suites.cec2017)

        # ML functions (require sklearn)
        try:
            from .test_functions.machine_learning import machine_learning_functions

            if machine_learning_functions:
                add_functions(machine_learning_functions)
        except ImportError:
            pass

        self._functions = functions

    def __iter__(self) -> Iterator[Type["BaseTestFunction"]]:
        self._ensure_initialized()
        return iter(self._functions)

    def __len__(self) -> int:
        self._ensure_initialized()
        return len(self._functions)

    def __contains__(self, item: Type["BaseTestFunction"]) -> bool:
        self._ensure_initialized()
        return item in self._functions

    def __getitem__(self, index: int) -> Type["BaseTestFunction"]:
        self._ensure_initialized()
        return self._functions[index]

    def filter(self, **criteria: Any) -> Collection:
        self._ensure_initialized()
        return super().filter(**criteria)

    def search(self, query: str) -> Collection:
        self._ensure_initialized()
        return super().search(query)

    def describe(self) -> str:
        self._ensure_initialized()
        return super().describe()

    def categories(self) -> List[str]:
        self._ensure_initialized()
        return super().categories()

    @property
    def names(self) -> List[str]:
        self._ensure_initialized()
        return super().names

    def _get_predefined(self, name: str) -> Collection:
        """Get a predefined collection by name (cached)."""
        if name not in self._predefined_cache:
            from .presets import suites as _suites

            preset_list = getattr(_suites, name)
            self._predefined_cache[name] = Collection(preset_list, name=name)
        return self._predefined_cache[name]

    @property
    def quick(self) -> Collection:
        """Quick sanity check (5 functions)."""
        return self._get_predefined("quick")

    @property
    def standard(self) -> Collection:
        """Academic comparison (15 functions)."""
        return self._get_predefined("standard")

    @property
    def bbob(self) -> Collection:
        """Full COCO/BBOB benchmark (24 functions)."""
        return self._get_predefined("bbob")

    @property
    def cec2014(self) -> Collection:
        """CEC 2014 competition functions (30 functions)."""
        return self._get_predefined("cec2014")

    @property
    def cec2017(self) -> Collection:
        """CEC 2017 competition functions (10 functions)."""
        return self._get_predefined("cec2017")

    @property
    def engineering(self) -> Collection:
        """Constrained engineering problems (5 functions)."""
        return self._get_predefined("engineering")

    @property
    def algebraic_2d(self) -> Collection:
        """All 2D algebraic functions (18 functions)."""
        return self._get_predefined("algebraic_2d")

    @property
    def algebraic_nd(self) -> Collection:
        """N-dimensional scalable functions (5 functions)."""
        return self._get_predefined("algebraic_nd")


# =============================================================================
# Module-level singleton instance
# =============================================================================

collection = _CollectionSingleton()
