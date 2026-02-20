"""Utility for checking optional dependencies at runtime."""

import importlib.util


def check_dependency(package: str, extras: str) -> None:
    """Check if an optional dependency is available.

    Parameters
    ----------
    package : str
        The package name to check (e.g. "lightgbm", "tensorflow").
    extras : str
        The pip extras name for the install hint (e.g. "ml", "images").

    Raises
    ------
    ImportError
        If the package is not installed, with install instructions.
    """
    if importlib.util.find_spec(package) is None:
        raise ImportError(
            f"This function requires '{package}'. " f"Install with: pip install surfaces[{extras}]"
        )
