# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utility functions for working with presets."""

import inspect

from .suites import (
    algebraic_2d,
    algebraic_nd,
    bbob,
    cec2014,
    cec2017,
    engineering,
    quick,
    standard,
)

_PRESETS = {
    "quick": quick,
    "standard": standard,
    "algebraic_2d": algebraic_2d,
    "algebraic_nd": algebraic_nd,
    "bbob": bbob,
    "cec2014": cec2014,
    "cec2017": cec2017,
    "engineering": engineering,
}


def instantiate(preset: list, n_dim: int = 10) -> list:
    """Instantiate all functions in a preset.

    Handles the complexity of presets with mixed function types
    (fixed-dimension vs scalable). Fixed-dimension functions are instantiated
    directly, while scalable functions receive the n_dim parameter.

    Parameters
    ----------
    preset : list
        List of function classes (e.g., quick, standard, bbob).
    n_dim : int, default=10
        Number of dimensions for scalable functions.

    Returns
    -------
    list
        List of instantiated function objects.

    Examples
    --------
    >>> from surfaces.presets import quick, instantiate
    >>> functions = instantiate(quick, n_dim=10)
    >>> for func in functions:
    ...     result = optimizer.minimize(func)

    >>> from surfaces.presets import bbob, instantiate
    >>> functions = instantiate(bbob, n_dim=20)
    """
    instances = []
    for FuncClass in preset:
        sig = inspect.signature(FuncClass.__init__)
        params = sig.parameters

        # Check if n_dim is a required parameter (no default value)
        if "n_dim" in params:
            param = params["n_dim"]
            if param.default is inspect.Parameter.empty:
                instances.append(FuncClass(n_dim=n_dim))
            else:
                instances.append(FuncClass())
        else:
            instances.append(FuncClass())

    return instances


def get(name: str) -> list:
    """Get a preset by name.

    Parameters
    ----------
    name : str
        Name of the preset: 'quick', 'standard', 'algebraic_2d', 'algebraic_nd',
        'bbob', 'cec2014', 'cec2017', or 'engineering'.

    Returns
    -------
    list
        List of function classes in the preset.

    Raises
    ------
    ValueError
        If the preset name is not recognized.

    Examples
    --------
    >>> from surfaces.presets import get
    >>> preset = get('quick')
    >>> for FuncClass in preset:
    ...     func = FuncClass(n_dim=5)
    """
    if name not in _PRESETS:
        available = ", ".join(sorted(_PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return _PRESETS[name]


def list_presets() -> dict:
    """List all available presets with their sizes.

    Returns
    -------
    dict
        Dictionary mapping preset names to number of functions.

    Examples
    --------
    >>> from surfaces.presets import list_presets
    >>> for name, count in list_presets().items():
    ...     print(f"{name}: {count} functions")
    """
    return {name: len(preset) for name, preset in _PRESETS.items()}
