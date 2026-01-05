# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Pipeline for applying multiple modifiers in sequence."""

from typing import Any, Dict, List, Optional, Type, TypeVar

from ._base_modifier import BaseModifier

T = TypeVar("T", bound=BaseModifier)


class ModifierPipeline:
    """Container for applying multiple modifiers in sequence.

    Modifiers are applied in the order they appear in the list.
    This order matters for non-commutative operations (e.g., rotation
    before scaling produces different results than scaling before rotation).

    Parameters
    ----------
    modifiers : list of BaseModifier
        Modifiers to apply in sequence.

    Examples
    --------
    >>> from surfaces.modifiers import GaussianNoise
    >>> from surfaces.modifiers import DelayModifier, ModifierPipeline
    >>> pipeline = ModifierPipeline([
    ...     DelayModifier(delay=0.01),
    ...     GaussianNoise(sigma=0.1)
    ... ])
    >>> value = pipeline.apply(5.0, {"x0": 1.0}, {"evaluation_count": 0})
    """

    def __init__(self, modifiers: List[BaseModifier]):
        self.modifiers = modifiers

    def apply(self, value: float, params: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Apply all modifiers in order.

        Parameters
        ----------
        value : float
            The function value to modify.
        params : dict
            The input parameters.
        context : dict
            Context information passed to each modifier.

        Returns
        -------
        float
            The value after all modifiers have been applied.
        """
        for modifier in self.modifiers:
            value = modifier.apply(value, params, context)
        return value

    def reset(self) -> None:
        """Reset all modifiers in the pipeline."""
        for modifier in self.modifiers:
            modifier.reset()

    def get_by_type(self, modifier_type: Type[T]) -> Optional[T]:
        """Get the first modifier of a specific type.

        Parameters
        ----------
        modifier_type : type
            The modifier class to search for.

        Returns
        -------
        BaseModifier or None
            The first modifier matching the type, or None if not found.

        Examples
        --------
        >>> from surfaces.modifiers import GaussianNoise
        >>> noise = pipeline.get_by_type(GaussianNoise)
        """
        for modifier in self.modifiers:
            if isinstance(modifier, modifier_type):
                return modifier
        return None

    def get_all_by_type(self, modifier_type: Type[T]) -> List[T]:
        """Get all modifiers of a specific type.

        Parameters
        ----------
        modifier_type : type
            The modifier class to search for.

        Returns
        -------
        list of BaseModifier
            All modifiers matching the type (empty list if none found).

        Examples
        --------
        >>> from surfaces.modifiers import DelayModifier
        >>> delays = pipeline.get_all_by_type(DelayModifier)
        """
        return [modifier for modifier in self.modifiers if isinstance(modifier, modifier_type)]

    def __getitem__(self, idx: int) -> BaseModifier:
        """Get a modifier by index.

        Parameters
        ----------
        idx : int
            Index of the modifier.

        Returns
        -------
        BaseModifier
            The modifier at the given index.
        """
        return self.modifiers[idx]

    def __len__(self) -> int:
        """Get the number of modifiers in the pipeline.

        Returns
        -------
        int
            Number of modifiers.
        """
        return len(self.modifiers)

    def __repr__(self) -> str:
        """String representation of the pipeline.

        Returns
        -------
        str
            String representation.
        """
        modifier_names = [type(mod).__name__ for mod in self.modifiers]
        return f"ModifierPipeline({modifier_names})"
