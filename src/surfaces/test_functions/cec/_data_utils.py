# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utilities for loading CEC benchmark data files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Data package name for CEC benchmark data
_DATA_PACKAGE = "surfaces_cec_data"


def _get_local_data_path(dataset: str, filename: str) -> Optional[Path]:
    """Check if file exists in the local data package directory (for development)."""
    # This file is at src/surfaces/test_functions/cec/_data_utils.py
    # Need 5 parents to get to repo root
    repo_root = Path(__file__).parent.parent.parent.parent.parent
    local_path = (
        repo_root
        / "data-packages"
        / "surfaces-cec-data"
        / "src"
        / "surfaces_cec_data"
        / dataset
        / filename
    )
    if local_path.exists():
        return local_path
    return None


def _get_installed_data_path(dataset: str, filename: str) -> Optional[Path]:
    """Get file path from the installed surfaces-cec-data package."""
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import as_file, files

            resource = files(_DATA_PACKAGE).joinpath(dataset, filename)
            try:
                with as_file(resource) as path:
                    if path.exists():
                        return path
            except (TypeError, FileNotFoundError):
                return None
        else:
            # Python 3.8 fallback
            try:
                from importlib_resources import as_file, files

                resource = files(_DATA_PACKAGE).joinpath(dataset, filename)
                with as_file(resource) as path:
                    if path.exists():
                        return path
            except ImportError:
                import importlib.resources as pkg_resources

                try:
                    with pkg_resources.path(f"{_DATA_PACKAGE}.{dataset}", filename) as path:
                        if path.exists():
                            return path
                except (ModuleNotFoundError, FileNotFoundError, TypeError):
                    return None
    except ModuleNotFoundError:
        return None
    return None


def _is_data_package_installed() -> bool:
    """Check if the surfaces-cec-data package is installed."""
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files

            files(_DATA_PACKAGE)
        else:
            try:
                from importlib_resources import files

                files(_DATA_PACKAGE)
            except ImportError:
                import importlib

                importlib.import_module(_DATA_PACKAGE)
        return True
    except ModuleNotFoundError:
        return False


def get_data_file(dataset: str, filename: str) -> Path:
    """Get path to a CEC data file.

    Checks local development directory first, then installed package.

    Parameters
    ----------
    dataset : str
        Name of the dataset (e.g., "cec2014", "cec2017").
    filename : str
        Name of the data file (e.g., "cec2014_data_dim10.npz").

    Returns
    -------
    Path
        Path to the data file.

    Raises
    ------
    ImportError
        If surfaces-cec-data is not installed and local files not found.
    FileNotFoundError
        If the specific file is not found.
    """
    # Check local development directory first
    local_path = _get_local_data_path(dataset, filename)
    if local_path is not None:
        return local_path

    # Check installed package
    installed_path = _get_installed_data_path(dataset, filename)
    if installed_path is not None:
        return installed_path

    # Neither found - provide helpful error
    if not _is_data_package_installed():
        raise ImportError(
            "CEC benchmark data files are not available.\n"
            "Install the data package with: pip install surfaces-cec-data\n"
            "Or install surfaces with CEC support: pip install surfaces[cec]"
        )
    else:
        raise FileNotFoundError(
            f"Data file not found: {dataset}/{filename}\n"
            "The surfaces-cec-data package may be outdated. "
            "Try: pip install -U surfaces-cec-data"
        )
