#!/usr/bin/env python3
"""Generate visualization assets for documentation.

This generator creates surface plots, contour plots, and thumbnails
for 2D test functions. Requires plotly and kaleido for static export.

Output Files
------------
docs/source/_generated/plots/
    {function_name}_surface.png
    {function_name}_contour.png
    {function_name}_thumb.png
    gallery.rst

Usage
-----
    python -m docs._generators.generate_plots

Requirements
------------
    pip install plotly kaleido

Note
----
This generator is optional. If plotly/kaleido are not installed,
it will skip without error.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

from . import extract_metadata, get_all_test_functions, get_function_hash
from .config import (
    DEFAULT_COLORSCALE,
    PLOT_HEIGHT,
    PLOT_RESOLUTION,
    PLOT_WIDTH,
    PLOTS_CACHE_FILE,
    PLOTS_DIR,
    THUMBNAIL_HEIGHT,
    THUMBNAIL_RESOLUTION,
    THUMBNAIL_WIDTH,
)

# Suppress warnings during plot generation
warnings.filterwarnings("ignore")


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        print("  Skipping: plotly not installed")
        print("  Install with: pip install plotly kaleido")
        return False

    try:
        import kaleido  # noqa: F401
    except ImportError:
        print("  Skipping: kaleido not installed")
        print("  Install with: pip install kaleido")
        return False

    return True


def load_cache() -> Dict[str, str]:
    """Load the plot generation cache."""
    if PLOTS_CACHE_FILE.exists():
        try:
            return json.loads(PLOTS_CACHE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_cache(cache: Dict[str, str]) -> None:
    """Save the plot generation cache."""
    try:
        PLOTS_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except IOError as e:
        print(f"  Warning: Could not save cache: {e}")


def should_regenerate(func_class: Type, cache: Dict[str, str]) -> bool:
    """Check if plots need to be regenerated for this function."""
    name = func_class.__name__
    current_hash = get_function_hash(func_class)
    cached_hash = cache.get(name)

    if current_hash != cached_hash:
        return True

    # Also check if output files exist
    meta = extract_metadata(func_class)
    internal_name = meta["internal_name"]

    surface_path = PLOTS_DIR / f"{internal_name}_surface.png"
    contour_path = PLOTS_DIR / f"{internal_name}_contour.png"
    thumb_path = PLOTS_DIR / f"{internal_name}_thumb.png"

    return not all([surface_path.exists(), contour_path.exists(), thumb_path.exists()])


def evaluate_function_grid(
    func_class: Type,
    resolution: int = PLOT_RESOLUTION,
) -> Optional[tuple]:
    """
    Evaluate a 2D function on a grid.

    Returns
    -------
    tuple or None
        (X, Y, Z, bounds) arrays or None if evaluation fails.
    """
    meta = extract_metadata(func_class)

    # Only process 2D functions
    if meta["n_dim"] != 2:
        return None

    # Create function instance
    try:
        func = func_class()
    except Exception as e:
        print(f"  Warning: Could not instantiate {func_class.__name__}: {e}")
        return None

    # Get bounds
    bounds = meta["default_bounds"]
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        x_min, x_max = bounds
        y_min, y_max = bounds
    else:
        x_min, x_max = -5, 5
        y_min, y_max = -5, 5

    # Create mesh grid
    x = np.linspace(x_min, x_max, resolution)
    y = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            try:
                Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})
            except Exception:
                Z[i, j] = np.nan

    return X, Y, Z, (x_min, x_max, y_min, y_max)


def generate_surface_plot(
    func_class: Type,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    colorscale: str = DEFAULT_COLORSCALE,
) -> Optional[Path]:
    """
    Generate 3D surface plot for a 2D function.

    Parameters
    ----------
    func_class : type
        Test function class.
    X, Y, Z : ndarray
        Mesh grid arrays from evaluate_function_grid.
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    Path or None
        Path to generated image, or None if generation fails.
    """
    import plotly.graph_objects as go

    meta = extract_metadata(func_class)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    len=0.7,
                ),
            )
        ]
    )

    fig.update_layout(
        title=dict(
            text=meta["display_name"],
            x=0.5,
            font=dict(size=16),
        ),
        scene=dict(
            xaxis_title="x0",
            yaxis_title="x1",
            zaxis_title="f(x)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
            ),
        ),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=0, r=0, t=50, b=0),
    )

    output_path = PLOTS_DIR / f"{meta['internal_name']}_surface.png"
    try:
        fig.write_image(str(output_path), scale=2)
        return output_path
    except Exception as e:
        print(f"  Warning: Could not save {output_path}: {e}")
        return None


def generate_contour_plot(
    func_class: Type,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    colorscale: str = DEFAULT_COLORSCALE,
) -> Optional[Path]:
    """
    Generate 2D contour plot for a 2D function.

    Parameters
    ----------
    func_class : type
        Test function class.
    X, Y, Z : ndarray
        Mesh grid arrays from evaluate_function_grid.
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    Path or None
        Path to generated image, or None if generation fails.
    """
    import plotly.graph_objects as go

    meta = extract_metadata(func_class)

    # Extract 1D arrays for contour (take first row/column)
    x = X[0, :]
    y = Y[:, 0]

    fig = go.Figure(
        data=[
            go.Contour(
                x=x,
                y=y,
                z=Z,
                colorscale=colorscale,
                contours=dict(
                    showlabels=True,
                    labelfont=dict(size=10, color="white"),
                ),
                colorbar=dict(
                    thickness=15,
                    len=0.9,
                ),
            )
        ]
    )

    # Mark global minimum if known
    x_global = meta["x_global"]
    if x_global is not None:
        try:
            if hasattr(x_global, "__len__") and len(x_global) >= 2:
                fig.add_trace(
                    go.Scatter(
                        x=[float(x_global[0])],
                        y=[float(x_global[1])],
                        mode="markers",
                        marker=dict(size=12, color="red", symbol="star"),
                        name="Global minimum",
                        showlegend=True,
                    )
                )
        except (TypeError, ValueError):
            pass

    fig.update_layout(
        title=dict(
            text=meta["display_name"],
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title="x0",
        yaxis_title="x1",
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        margin=dict(l=60, r=60, t=50, b=60),
    )

    output_path = PLOTS_DIR / f"{meta['internal_name']}_contour.png"
    try:
        fig.write_image(str(output_path), scale=2)
        return output_path
    except Exception as e:
        print(f"  Warning: Could not save {output_path}: {e}")
        return None


def generate_thumbnail(
    func_class: Type,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
) -> Optional[Path]:
    """
    Generate small thumbnail for gallery view.

    Parameters
    ----------
    func_class : type
        Test function class.
    X, Y, Z : ndarray
        Mesh grid arrays from evaluate_function_grid.

    Returns
    -------
    Path or None
        Path to generated image, or None if generation fails.
    """
    import plotly.graph_objects as go

    meta = extract_metadata(func_class)

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale=DEFAULT_COLORSCALE,
                showscale=False,
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.0),
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        width=THUMBNAIL_WIDTH,
        height=THUMBNAIL_HEIGHT,
    )

    output_path = PLOTS_DIR / f"{meta['internal_name']}_thumb.png"
    try:
        fig.write_image(str(output_path), scale=2)
        return output_path
    except Exception as e:
        print(f"  Warning: Could not save {output_path}: {e}")
        return None


def generate_gallery_rst(functions_2d: List[Type]) -> Path:
    """
    Generate RST file with image gallery.

    Parameters
    ----------
    functions_2d : list
        List of 2D test function classes.

    Returns
    -------
    Path
        Path to generated gallery.rst file.
    """
    lines = [
        ".. _function_gallery:",
        "",
        "2D Function Gallery",
        "===================",
        "",
        "Visual overview of all 2D algebraic test functions. Click on any function",
        "to see its detailed documentation.",
        "",
        ".. grid:: 2 3 4 4",
        "   :gutter: 2",
        "",
    ]

    for func_class in sorted(functions_2d, key=lambda f: f.__name__):
        meta = extract_metadata(func_class)
        if meta["n_dim"] != 2:
            continue

        internal_name = meta["internal_name"]
        thumb_path = f"/_generated/plots/{internal_name}_thumb.png"
        surface_path = f"/_generated/plots/{internal_name}_surface.png"

        # Check if thumbnail exists
        if not (PLOTS_DIR / f"{internal_name}_thumb.png").exists():
            continue

        lines.extend(
            [
                "   .. grid-item-card::",
                f"      :img-top: {thumb_path}",
                f"      :link: {meta['module']}.{meta['name']}",
                "      :link-type: ref",
                "",
                f"      **{meta['display_name']}**",
                "",
            ]
        )

    output_path = PLOTS_DIR / "gallery.rst"
    output_path.write_text("\n".join(lines))

    return output_path


def generate_function_detail_rst(func_class: Type) -> Optional[Path]:
    """
    Generate RST file for a single function with plots.

    This creates a detail page showing both surface and contour plots
    for a single function.
    """
    meta = extract_metadata(func_class)
    if meta["n_dim"] != 2:
        return None

    internal_name = meta["internal_name"]
    surface_exists = (PLOTS_DIR / f"{internal_name}_surface.png").exists()
    contour_exists = (PLOTS_DIR / f"{internal_name}_contour.png").exists()

    if not (surface_exists or contour_exists):
        return None

    lines = [
        f".. _{internal_name}_plots:",
        "",
        f"{meta['display_name']} Plots",
        "=" * (len(meta["display_name"]) + 6),
        "",
    ]

    if surface_exists:
        lines.extend(
            [
                "Surface Plot",
                "------------",
                "",
                f".. image:: /_generated/plots/{internal_name}_surface.png",
                "   :alt: Surface plot",
                "   :align: center",
                "",
            ]
        )

    if contour_exists:
        lines.extend(
            [
                "Contour Plot",
                "------------",
                "",
                f".. image:: /_generated/plots/{internal_name}_contour.png",
                "   :alt: Contour plot",
                "   :align: center",
                "",
            ]
        )

    output_path = PLOTS_DIR / f"{internal_name}_detail.rst"
    output_path.write_text("\n".join(lines))

    return output_path


def main(force: bool = False):
    """
    Generate all plot assets.

    Parameters
    ----------
    force : bool
        If True, regenerate all plots regardless of cache.
    """
    print(f"Output directory: {PLOTS_DIR}")

    # Check dependencies
    if not check_dependencies():
        return

    # Load cache
    cache = {} if force else load_cache()

    # Get all 2D functions
    categories = get_all_test_functions()
    functions_2d = categories.get("algebraic_2d", [])

    print(f"Found {len(functions_2d)} 2D algebraic functions")

    generated_count = 0
    skipped_count = 0

    for func_class in functions_2d:
        meta = extract_metadata(func_class)

        # Check if we need to regenerate
        if not force and not should_regenerate(func_class, cache):
            skipped_count += 1
            continue

        print(f"  Generating plots for {meta['display_name']}...")

        # Evaluate function on grid
        grid_data = evaluate_function_grid(func_class)
        if grid_data is None:
            print(f"    Warning: Could not evaluate {func_class.__name__}")
            continue

        X, Y, Z, bounds = grid_data

        # Also evaluate at lower resolution for thumbnail
        thumb_data = evaluate_function_grid(func_class, THUMBNAIL_RESOLUTION)
        if thumb_data is None:
            X_thumb, Y_thumb, Z_thumb = X, Y, Z
        else:
            X_thumb, Y_thumb, Z_thumb, _ = thumb_data

        # Generate all three plot types
        surface_path = generate_surface_plot(func_class, X, Y, Z)
        contour_path = generate_contour_plot(func_class, X, Y, Z)
        thumb_path = generate_thumbnail(func_class, X_thumb, Y_thumb, Z_thumb)

        if surface_path and contour_path and thumb_path:
            # Update cache
            cache[func_class.__name__] = get_function_hash(func_class)
            generated_count += 1

            # Generate detail page
            generate_function_detail_rst(func_class)

    # Save updated cache
    save_cache(cache)

    # Generate gallery
    print("  Generating gallery.rst...")
    gallery_path = generate_gallery_rst(functions_2d)
    print(f"  Generated: {gallery_path}")

    print("\nPlot generation complete:")
    print(f"  Generated: {generated_count} functions")
    print(f"  Skipped (cached): {skipped_count} functions")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate plot assets")
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force regeneration of all plots",
    )
    args = parser.parse_args()

    main(force=args.force)
