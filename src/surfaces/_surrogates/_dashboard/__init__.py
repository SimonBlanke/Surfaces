# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""
Surrogate Dashboard - Management UI for ML test function surrogates.

This module provides a Streamlit-based dashboard for:
- Viewing all ML test functions and their surrogate status
- Training new or retraining existing surrogates
- Validating surrogate accuracy
- Tracking historical metrics

Usage:
    # Via module
    python -m surfaces._surrogates._dashboard

    # Via CLI (after installing with dashboard extras)
    surfaces-dashboard

    # Via Python
    from surfaces._surrogates._dashboard import run_dashboard
    run_dashboard()

Requirements:
    pip install surfaces[dashboard]
"""

from pathlib import Path


def run_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import sys

    app_path = Path(__file__).parent / "app.py"

    # Check if streamlit is installed
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("Streamlit is required for the dashboard.")
        print("Install it with: pip install surfaces[dashboard]")
        sys.exit(1)

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.headless", "true"]
    )


__all__ = ["run_dashboard"]
