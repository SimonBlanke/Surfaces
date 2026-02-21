"""ModifierAccessor: Sequence-like access to modifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from surfaces.modifiers import BaseModifier

    from .._base_test_function import BaseTestFunction


class ModifierAccessor(Sequence):
    """Namespaced access to modifier management.

    Implements the Sequence protocol for iteration and indexing.

    Parameters
    ----------
    func : BaseTestFunction
        The test function instance.
    """

    def __init__(self, func: "BaseTestFunction") -> None:
        self._func = func

    def add(self, modifier: "BaseModifier") -> None:
        """Add a modifier to the modifier chain."""
        self._func._modifiers.append(modifier)

    def remove(self, modifier: "BaseModifier") -> None:
        """Remove a modifier from the chain.

        Raises ValueError if the modifier is not found.
        """
        self._func._modifiers.remove(modifier)

    def clear(self) -> None:
        """Remove all modifiers."""
        self._func._modifiers.clear()

    def reset(self) -> None:
        """Reset all modifiers' internal state."""
        for modifier in self._func._modifiers:
            modifier.reset()

    def __getitem__(self, index):
        return self._func._modifiers[index]

    def __len__(self) -> int:
        return len(self._func._modifiers)

    def __iter__(self) -> Iterator:
        return iter(self._func._modifiers)

    def __repr__(self) -> str:
        return f"ModifierAccessor({len(self)} modifiers)"
