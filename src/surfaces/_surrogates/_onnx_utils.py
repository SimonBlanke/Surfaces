# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Utilities for loading ONNX surrogate model files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Primary package for ONNX surrogate models (surfaces-surrogates)
_ONNX_PACKAGE = "surfaces_surrogates.models"

# Legacy package for backwards compatibility during transition
_ONNX_PACKAGE_LEGACY = "surfaces_onnx_files"

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
    """Get file path from the installed surrogate models package."""
    for package_name in (_ONNX_PACKAGE, _ONNX_PACKAGE_LEGACY):
        try:
            from importlib.resources import as_file, files

            resource = files(package_name).joinpath(filename)
            try:
                with as_file(resource) as path:
                    if path.exists():
                        return path
            except (TypeError, FileNotFoundError):
                continue
        except ModuleNotFoundError:
            continue
    return None


def _is_onnx_package_installed() -> bool:
    """Check if a surrogate models package is installed.

    Returns True if either surfaces-surrogates or the legacy
    surfaces-onnx-files package is available.
    """
    from importlib.resources import files

    for package_name in (_ONNX_PACKAGE, _ONNX_PACKAGE_LEGACY):
        try:
            files(package_name)
            return True
        except ModuleNotFoundError:
            continue
    return False


def get_onnx_file(filename: str) -> Optional[Path]:
    """Get path to an ONNX model file.

    Checks in order:
    1. Local models directory (for newly trained models)
    2. Local data package directory (for development)
    3. Installed surfaces-surrogates package (surfaces_surrogates.models)
    4. Installed surfaces-onnx-files package (legacy fallback)

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
