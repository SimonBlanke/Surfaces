"""Generate Poincare disk model SVG logo.

Produces a hyperbolic geometry visualization using geodesic arcs in the
Poincare disk model. Geodesics are circular arcs that meet the boundary
circle at right angles.

Usage:
    python generate_hyperbolic.py
    python generate_hyperbolic.py --output my_logo.svg
"""

import argparse
import math
from pathlib import Path

# -- Configuration -----------------------------------------------------------
# Edit this dict to tweak the logo. Layers are drawn in order (first = behind).

CONFIG = {
    "viewbox_size": 400,
    "disk_radius": 180,
    "background_color": None,
    "border_color": "#9093bf",
    "border_width": 2,
    "layers": [
        # Secondary layers (background web, 36 boundary points)
        {
            "n_points": 36,
            "offset": 5,
            "stroke": "#a0c4f0",
            "width": 0.4,
            "opacity": 0.3,
        },
        {
            "n_points": 36,
            "offset": 11,
            "stroke": "#a0c4f0",
            "width": 0.4,
            "opacity": 0.3,
        },
        {
            "n_points": 36,
            "offset": 17,
            "stroke": "#2b8cdc",  # gfo
            "width": 0.3,
            "opacity": 0.70,
        },
        # Primary layers (structural, 24 boundary points)
        {
            "n_points": 24,
            "offset": 3,
            "stroke": "#88b8e8",
            "width": 1.0,
            "opacity": 0.50,
        },
        {
            "n_points": 24,
            "offset": 5,
            "stroke": "#88b8e8",
            "width": 0.85,
            "opacity": 0.65,
        },
        {
            "n_points": 24,
            "offset": 7,
            "stroke": "#90c0f0",
            "width": 0.7,
            "opacity": 0.50,
        },
        {
            "n_points": 24,
            "offset": 11,
            "stroke": "#33164f",  # hyperactive
            "width": 0.3,
            "opacity": 0.50,
        },
    ],
    # Emphasis rings (solid circles like the border, drawn on top)
    "rings": [
        # {"radius": 182, "stroke": "#70a8e0", "width": 1.5},
        {"radius": 55, "stroke": "#9093bf", "width": 1.5},
        {"radius": 12, "stroke": "#9093bf", "width": 1},
    ],
    "output_file": "logo.svg",
}


def boundary_points(n, radius, center):
    """Compute evenly-spaced points on the boundary circle.

    Parameters
    ----------
    n : int
        Number of points around the circle.
    radius : float
        Radius of the boundary circle.
    center : tuple of float
        (cx, cy) center of the disk.

    Returns
    -------
    list of tuple
        List of (x, y) coordinates.
    """
    cx, cy = center
    return [
        (
            cx + radius * math.cos(i * 2 * math.pi / n),
            cy + radius * math.sin(i * 2 * math.pi / n),
        )
        for i in range(n)
    ]


def geodesic_arc_svg(p1, p2, center, disk_radius):
    """Compute the SVG path string for a hyperbolic geodesic arc.

    Given two points on the boundary of a Poincare disk, the geodesic
    between them is an arc of a circle that intersects the boundary at
    right angles. The arc center lies along the bisector of the two
    boundary points, at distance d = R / cos(half_da) from the disk
    center, and the arc radius is rg = R * tan(half_da).

    Parameters
    ----------
    p1 : tuple of float
        (x, y) of the first boundary point.
    p2 : tuple of float
        (x, y) of the second boundary point.
    center : tuple of float
        (cx, cy) of the disk center.
    disk_radius : float
        Radius of the Poincare disk.

    Returns
    -------
    str
        SVG path data string (e.g. "M x1,y1 A rx,ry 0 0,0 x2,y2").
    """
    cx, cy = center
    x1, y1 = p1
    x2, y2 = p2

    # Angles of each point relative to the disk center
    a1 = math.atan2(y1 - cy, x1 - cx)
    a2 = math.atan2(y2 - cy, x2 - cx)

    # Normalize the angular separation to (0, pi). The offset-based
    # construction guarantees the two points are never diametrically
    # opposite (offset < n/2), so da is always strictly between 0 and pi.
    da = (a2 - a1) % (2 * math.pi)
    if da > math.pi:
        da = 2 * math.pi - da

    half_da = da / 2
    arc_radius = disk_radius * math.tan(half_da)

    # Determine the SVG sweep flag. In SVG's Y-down coordinate system,
    # we pick the arc that bows toward the disk center. The cross product
    # of (P2 - P1) and (DiskCenter - P1) tells us which side of the
    # P1->P2 line the disk center lies on. In Y-down coords, a positive
    # cross product means the center is to the visual left of P1->P2,
    # so sweep=0 (counterclockwise on screen) curves toward it.
    dx12 = x2 - x1
    dy12 = y2 - y1
    dxc = cx - x1
    dyc = cy - y1
    cross = dx12 * dyc - dy12 * dxc
    sweep = 0 if cross > 0 else 1

    return (
        f"M {x1:.2f},{y1:.2f} " f"A {arc_radius:.2f},{arc_radius:.2f} 0 0,{sweep} {x2:.2f},{y2:.2f}"
    )


def generate_svg(config):
    """Build the complete SVG string from the configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys: viewbox_size, disk_radius,
        background_color, border_color, border_width, layers.

    Returns
    -------
    str
        Complete SVG document as a string.
    """
    size = config["viewbox_size"]
    radius = config["disk_radius"]
    cx = size / 2
    cy = size / 2
    center = (cx, cy)

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {size} {size}" '
        f'width="{size}" height="{size}">'
    )

    # Clip path to keep arcs inside the disk
    lines.append("  <defs>")
    lines.append('    <clipPath id="disk-clip">')
    lines.append(f'      <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{radius}"/>')
    lines.append("    </clipPath>")
    lines.append("  </defs>")
    lines.append("")

    # Background fill (optional)
    if config["background_color"]:
        lines.append(
            f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{radius}" '
            f'fill="{config["background_color"]}" stroke="none"/>'
        )
        lines.append("")

    # Clipped group for all geodesic arcs
    lines.append('  <g clip-path="url(#disk-clip)">')

    for layer in config["layers"]:
        n = layer["n_points"]
        offset = layer["offset"]
        stroke = layer["stroke"]
        width = layer["width"]
        opacity = layer["opacity"]
        points = boundary_points(n, radius, center)

        for i in range(n):
            j = (i + offset) % n
            path_data = geodesic_arc_svg(points[i], points[j], center, radius)
            lines.append(
                f'    <path d="{path_data}" fill="none" '
                f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"/>'
            )

    lines.append("  </g>")
    lines.append("")

    # Emphasis rings (solid circles to pronounce natural ring patterns)
    for ring in config.get("rings", []):
        lines.append(
            f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{ring["radius"]}" '
            f'fill="none" stroke="{ring["stroke"]}" '
            f'stroke-width="{ring["width"]}"/>'
        )

    # Border ring
    lines.append(
        f'  <circle cx="{cx:.0f}" cy="{cy:.0f}" r="{radius}" '
        f'fill="none" stroke="{config["border_color"]}" '
        f'stroke-width="{config["border_width"]}"/>'
    )
    lines.append("")
    lines.append("</svg>")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a Poincare disk model SVG logo.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output SVG file path. Defaults to the config value.",
    )
    args = parser.parse_args()

    output_path = args.output or CONFIG["output_file"]
    # Resolve relative paths against this script's directory
    output = Path(output_path)
    if not output.is_absolute():
        output = Path(__file__).parent / output

    svg_content = generate_svg(CONFIG)
    output.write_text(svg_content, encoding="utf-8")
    print(f"Generated: {output}")


if __name__ == "__main__":
    main()
