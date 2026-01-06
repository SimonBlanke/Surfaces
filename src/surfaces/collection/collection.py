# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Collection class for filtering and combining test functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, List, Optional, Type

from .utils import get_category, get_merged_spec, get_n_dim

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction


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
                result = [f for f in result if get_category(f) == value]
            elif key == "n_dim":
                result = [f for f in result if get_n_dim(f) == value]
            else:
                # Filter by spec property
                from .utils import get_spec_value

                result = [f for f in result if get_spec_value(f, key) == value]

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
            category = get_category(func_cls)
            n_dim = get_n_dim(func_cls)
            dim_str = str(n_dim) if n_dim else "N"

            spec = get_merged_spec(func_cls)
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
        cats = set(get_category(f) for f in self._functions)
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
