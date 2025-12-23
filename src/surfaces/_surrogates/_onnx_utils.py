# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utilities for loading ONNX surrogate model files."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

# Data package name for ONNX surrogate models
_ONNX_PACKAGE = "surfaces_onnx_files"

# Local models directory (for newly trained models)
_LOCAL_MODELS_DIR = Path(__file__).parent / "models"


def _get_trained_model_path(filename: str) -> Optional[Path]:
    """Check if file exists in the local models directory (for newly trained models)."""
    local_path = _LOCAL_MODELS_DIR / filename
    if local_path.exists():
        return local_path
    return None


def _get_local_onnx_path(filename: str) -> Optional[Path]:
    """Check if file exists in the local data package directory (for development)."""
    # This file is at src/surfaces/_surrogates/_onnx_utils.py
    # Need 4 parents to get to repo root
    repo_root = Path(__file__).parent.parent.parent.parent
    local_path = (
        repo_root
        / "data-packages"
        / "surfaces-onnx-files"
        / "src"
        / "surfaces_onnx_files"
        / filename
    )
    if local_path.exists():
        return local_path
    return None


def _get_installed_onnx_path(filename: str) -> Optional[Path]:
    """Get file path from the installed surfaces-onnx-files package."""
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import as_file, files

            resource = files(_ONNX_PACKAGE).joinpath(filename)
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

                resource = files(_ONNX_PACKAGE).joinpath(filename)
                with as_file(resource) as path:
                    if path.exists():
                        return path
            except ImportError:
                import importlib.resources as pkg_resources

                try:
                    with pkg_resources.path(_ONNX_PACKAGE, filename) as path:
                        if path.exists():
                            return path
                except (ModuleNotFoundError, FileNotFoundError, TypeError):
                    return None
    except ModuleNotFoundError:
        return None
    return None


def _is_onnx_package_installed() -> bool:
    """Check if the surfaces-onnx-files package is installed."""
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files

            files(_ONNX_PACKAGE)
        else:
            try:
                from importlib_resources import files

                files(_ONNX_PACKAGE)
            except ImportError:
                import importlib

                importlib.import_module(_ONNX_PACKAGE)
        return True
    except ModuleNotFoundError:
        return False


def get_onnx_file(filename: str) -> Optional[Path]:
    """Get path to an ONNX model file.

    Checks in order:
    1. Local models directory (for newly trained models)
    2. Local data package directory (for development)
    3. Installed surfaces-onnx-files package

    Parameters
    ----------
    filename : str
        Name of the ONNX file (e.g., "k_neighbors_regressor.onnx").

    Returns
    -------
    Path or None
        Path to the file if found, None otherwise.
    """
    # Check local models directory first (for newly trained models)
    trained_path = _get_trained_model_path(filename)
    if trained_path is not None:
        return trained_path

    # Check local data package directory (for development)
    local_path = _get_local_onnx_path(filename)
    if local_path is not None:
        return local_path

    # Check installed package
    installed_path = _get_installed_onnx_path(filename)
    if installed_path is not None:
        return installed_path

    return None


def get_surrogate_model_path(function_name: str) -> Optional[Path]:
    """Get path to a pre-trained surrogate model.

    Parameters
    ----------
    function_name : str
        Name of the function (e.g., "k_neighbors_classifier").

    Returns
    -------
    Path or None
        Path to the ONNX file if it exists, None otherwise.
    """
    return get_onnx_file(f"{function_name}.onnx")


def get_metadata_path(function_name: str) -> Optional[Path]:
    """Get path to a surrogate model's metadata file.

    Parameters
    ----------
    function_name : str
        Name of the function (e.g., "k_neighbors_classifier").

    Returns
    -------
    Path or None
        Path to the .meta.json file if it exists, None otherwise.
    """
    return get_onnx_file(f"{function_name}.onnx.meta.json")


def get_validity_model_path(function_name: str) -> Optional[Path]:
    """Get path to a surrogate model's validity model.

    Parameters
    ----------
    function_name : str
        Name of the function (e.g., "k_neighbors_classifier").

    Returns
    -------
    Path or None
        Path to the .validity.onnx file if it exists, None otherwise.
    """
    return get_onnx_file(f"{function_name}.validity.onnx")
