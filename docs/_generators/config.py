"""Configuration constants for documentation generators.

This module defines paths and settings used by all generator scripts.
All paths are resolved relative to the project root.
"""

from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Documentation directories
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_SOURCE = DOCS_DIR / "source"

# Generated content output directory
GENERATED_DIR = DOCS_SOURCE / "_generated"

# Subdirectories for different generated content types
API_DIR = GENERATED_DIR / "api"
CATALOGS_DIR = GENERATED_DIR / "catalogs"
PLOTS_DIR = GENERATED_DIR / "plots"
DIAGRAMS_DIR = GENERATED_DIR / "diagrams"

# Source code directory
SRC_DIR = PROJECT_ROOT / "src"

# =============================================================================
# Plot Configuration
# =============================================================================

# Full-size plot dimensions
PLOT_WIDTH = 800
PLOT_HEIGHT = 600
PLOT_DPI = 150

# Thumbnail dimensions for gallery
THUMBNAIL_WIDTH = 300
THUMBNAIL_HEIGHT = 225

# Default colorscale for surface plots
DEFAULT_COLORSCALE = "Viridis"

# Grid resolution for function evaluation
PLOT_RESOLUTION = 100
THUMBNAIL_RESOLUTION = 50

# =============================================================================
# Cache Configuration
# =============================================================================

# Cache file for tracking which plots need regeneration
PLOTS_CACHE_FILE = PLOTS_DIR / ".cache.json"

# =============================================================================
# Category Display Names
# =============================================================================

CATEGORY_DISPLAY_NAMES = {
    "algebraic_1d": "1D Algebraic",
    "algebraic_2d": "2D Algebraic",
    "algebraic_nd": "N-D Algebraic",
    "ml_tabular_classification": "ML Classification (Tabular)",
    "ml_tabular_regression": "ML Regression (Tabular)",
    "ml_image_classification": "ML Classification (Image)",
    "ml_timeseries_classification": "ML Classification (Time Series)",
    "ml_timeseries_forecasting": "ML Forecasting (Time Series)",
    "engineering": "Engineering Design",
    "bbob": "BBOB Functions",
    "cec": "CEC Functions",
}

CATEGORY_DESCRIPTIONS = {
    "algebraic_1d": "Single-variable mathematical test functions.",
    "algebraic_2d": "Two-dimensional mathematical test functions with known optima.",
    "algebraic_nd": "Scalable n-dimensional functions that work with any number of variables.",
    "ml_tabular_classification": "Hyperparameter optimization landscapes for tabular classification models.",
    "ml_tabular_regression": "Hyperparameter optimization landscapes for tabular regression models.",
    "ml_image_classification": "Hyperparameter optimization landscapes for image classification models.",
    "ml_timeseries_classification": "Hyperparameter optimization landscapes for time series classification.",
    "ml_timeseries_forecasting": "Hyperparameter optimization landscapes for time series forecasting.",
    "engineering": "Real-world constrained engineering design optimization problems.",
    "bbob": "Black-Box Optimization Benchmarking (BBOB) test suite functions.",
    "cec": "Congress on Evolutionary Computation (CEC) benchmark functions.",
}

# =============================================================================
# Directory Initialization
# =============================================================================


def ensure_directories():
    """Create all output directories if they don't exist."""
    for directory in [API_DIR, CATALOGS_DIR, PLOTS_DIR, DIAGRAMS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


# Ensure directories exist on import
ensure_directories()
