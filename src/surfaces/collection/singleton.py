# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Singleton collection containing all test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Type

from .collection import Collection

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


class _CollectionSingleton(Collection):
    """Singleton collection containing all test functions.

    This class extends Collection with lazy-loaded predefined subsets
    accessible as properties (quick, standard, bbob, etc.).

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
    >>> collection.constrained                # 5 constrained problems
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
        from ..test_functions.algebraic import algebraic_functions

        add_functions(algebraic_functions)

        # BBOB functions (always available)
        from ..test_functions.benchmark.bbob import bbob_functions

        add_functions(bbob_functions)

        # Constrained functions (always available)
        from ..test_functions.algebraic.constrained import constrained_functions

        add_functions(constrained_functions)

        # CEC functions (require cec data package)
        try:
            from ..test_functions.benchmark.cec.cec2013 import cec2013_functions
            from ..test_functions.benchmark.cec.cec2014 import cec2014_functions
            from ..test_functions.benchmark.cec.cec2017 import cec2017_functions

            add_functions(cec2013_functions)
            add_functions(cec2014_functions)
            add_functions(cec2017_functions)
        except ImportError:
            pass  # CEC data package not installed

        # ML functions (require sklearn)
        try:
            from ..test_functions.machine_learning import machine_learning_functions

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

    def __repr__(self) -> str:
        self._ensure_initialized()
        from . import suites

        suite_info = ", ".join(
            f"{name}({len(getattr(suites, name))})" for name in suites.SUITE_NAMES
        )
        return f"Collection({len(self)} functions)\nSuites: {suite_info}"

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
            from . import suites

            preset_list = getattr(suites, name)
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
    def constrained(self) -> Collection:
        """Constrained engineering problems (5 functions)."""
        return self._get_predefined("constrained")

    @property
    def algebraic_2d(self) -> Collection:
        """All 2D algebraic functions (18 functions)."""
        return self._get_predefined("algebraic_2d")

    @property
    def algebraic_nd(self) -> Collection:
        """N-dimensional scalable functions (5 functions)."""
        return self._get_predefined("algebraic_nd")

    def show(self) -> Dict[str, int]:
        """Show all available suites with their sizes.

        Returns
        -------
        dict
            Dictionary mapping suite names to number of functions.

        Examples
        --------
        >>> from surfaces import collection
        >>> collection.show()
        {'quick': 5, 'standard': 15, ...}
        """
        from . import suites

        return {name: len(getattr(suites, name)) for name in suites.SUITE_NAMES}
