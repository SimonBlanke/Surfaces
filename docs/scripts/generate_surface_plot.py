#!/usr/bin/env python3
"""Generate surface plots for the README."""

import re

import numpy as np
import plotly.graph_objects as go
from PIL import Image

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    CrossInTrayFunction,
    DropWaveFunction,
    EggholderFunction,
    HimmelblausFunction,
    RastriginFunction,
)

# Test functions to visualize (tighter bounds for some)
FUNCTIONS = [
    ("ackley", AckleyFunction(), (-3, 3)),
    ("himmelblau", HimmelblausFunction(), (-5, 5)),
    ("drop_wave", DropWaveFunction(), (-3, 3)),
    ("eggholder", EggholderFunction(), (-512, 512)),
    ("cross_in_tray", CrossInTrayFunction(), (-5, 5)),
    ("rastrigin", RastriginFunction(n_dim=2), (-5.12, 5.12)),
]

OUTPUT_DIR = "../source/_static"


def generate_surface_plot(name, func, bounds, resolution=150):
    """Generate a surface plot SVG for a test function."""
    min_val, max_val = bounds

    # Create mesh grid
    x = np.linspace(min_val, max_val, resolution)
    y = np.linspace(min_val, max_val, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    points = np.column_stack([X.ravel(), Y.ravel()])
    Z = func._batch_objective(points).reshape(X.shape)

    # Create figure with Jet_r colorscale (red = low/optimal, blue = high)
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                colorscale="Jet_r",
                showscale=False,
                lighting={
                    "ambient": 0.6,
                    "diffuse": 0.8,
                    "specular": 0.2,
                    "roughness": 0.5,
                },
            )
        ]
    )

    # Layout
    fig.update_layout(
        scene={
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "zaxis": {"visible": False},
            "bgcolor": "rgba(0,0,0,0)",
            "camera": {
                "eye": {"x": 1.3, "y": 1.3, "z": 0.6},
            },
            "aspectmode": "manual",
            "aspectratio": {"x": 1, "y": 1, "z": 0.5},
            "domain": {"x": [0, 1], "y": [0, 1]},
        },
        margin={"l": 0, "r": 0, "t": 0, "b": 0, "pad": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        width=900,
        height=600,
        autosize=False,
    )

    # Save temp PNG for bbox calculation
    temp_png = f"/tmp/{name}_temp.png"
    fig.write_image(temp_png, scale=2)

    # Get crop bounds from PNG
    img = Image.open(temp_png).convert("RGBA")
    bbox = img.getbbox()

    if bbox:
        padding = 20
        left = max(0, bbox[0] - padding)
        top = max(0, bbox[1] - padding)
        right = min(img.width, bbox[2] + padding)
        bottom = min(img.height, bbox[3] + padding)
    else:
        left, top, right, bottom = 0, 0, img.width, img.height

    # Save and crop SVG
    temp_svg = f"/tmp/{name}_temp.svg"
    fig.write_image(temp_svg)

    with open(temp_svg, "r") as f:
        svg_content = f.read()

    # Calculate crop for SVG (original was 1800x1200 at scale=2)
    orig_width, orig_height = 1800, 1200
    svg_width, svg_height = 900, 600

    new_x = (left / orig_width) * svg_width
    new_y = (top / orig_height) * svg_height
    new_w = ((right - left) / orig_width) * svg_width
    new_h = ((bottom - top) / orig_height) * svg_height

    svg_content = re.sub(r'width="[^"]*"', f'width="{new_w:.0f}"', svg_content, count=1)
    svg_content = re.sub(r'height="[^"]*"', f'height="{new_h:.0f}"', svg_content, count=1)
    svg_content = re.sub(
        r'viewBox="[^"]*"',
        f'viewBox="{new_x:.1f} {new_y:.1f} {new_w:.1f} {new_h:.1f}"',
        svg_content,
        count=1,
    )

    output_path = f"{OUTPUT_DIR}/{name}_surface.svg"
    with open(output_path, "w") as f:
        f.write(svg_content)

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    for name, func, bounds in FUNCTIONS:
        print(f"Generating {name}...")
        generate_surface_plot(name, func, bounds)

    print("\nDone! Generated 6 SVG files.")
