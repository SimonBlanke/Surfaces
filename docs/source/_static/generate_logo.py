#!/usr/bin/env python3
"""
Generate wireframe mesh sphere logos for the Surfaces project.

Concept: A solid filled sphere with a cutaway section that reveals the
wireframe mesh underneath. At a glance it reads as a bold circle/sphere.
On closer inspection, the cutaway shows the mathematical surface structure.

Usage:
    python generate_logo.py

Output:
    surfaces_logo.svg      (light mode - dark blue on transparent)
    surfaces_logo_dark.svg  (dark mode - light blue on transparent)
"""

import math
import os

# =============================================================================
# Configuration
# =============================================================================

# Canvas
WIDTH = 500
HEIGHT = 500
CENTER_X = WIDTH / 2
CENTER_Y = HEIGHT / 2
BASE_RADIUS = 200

# Mesh density (moderate - only visible through cutaway)
N_LATITUDE = 12
N_LONGITUDE = 18
POINTS_PER_LINE = 100

# Surface distortion
DISTORTION_MODES = [
    (3, 2, 0.14),
    (2, 3, 0.08),
]

# Viewing angle
TILT_X = math.radians(25)
TILT_Z = math.radians(15)

# Wireframe styling
STROKE_WIDTH_FRONT = 2.0
STROKE_WIDTH_BACK = 0.5
OPACITY_FRONT = 0.85
OPACITY_BACK = 0.10

# Cutaway definition (math convention: 0=right, counter-clockwise)
# The cutaway reveals the upper-right portion of the sphere
CUTAWAY_ANGLE_A = math.radians(128)  # point on circle, upper-left area
CUTAWAY_ANGLE_B = math.radians(-48)  # point on circle, lower-right area
# Bezier control points for the cutaway boundary curve (as fraction of radius)
# These shape the S-curve that separates solid from wireframe
CUTAWAY_CP1 = (0.15, 0.40)  # control point 1 (x_frac, y_frac from center)
CUTAWAY_CP2 = (0.20, -0.15)  # control point 2

# Cutaway edge styling
CUTAWAY_EDGE_WIDTH = 1.5

# Solid fill gradient (radial, to suggest 3D curvature)
# Defined as (offset, opacity) pairs
GRADIENT_STOPS = [
    (0.0, 0.75),  # center: slightly lighter
    (0.7, 0.88),  # mid: medium
    (1.0, 0.95),  # edge: darkest
]

# Colors
COLORS_LIGHT = {
    "fill": "#0008e6",
    "wireframe": "#0008e6",
    "edge": "#0008e6",
    "gradient_base": "#0008e6",
}
COLORS_DARK = {
    "fill": "#80b4ff",
    "wireframe": "#80b4ff",
    "edge": "#80b4ff",
    "gradient_base": "#80b4ff",
}


# =============================================================================
# Math helpers
# =============================================================================


def distorted_radius(theta, phi, base_r, modes):
    """Compute radius with spherical harmonic-like distortion."""
    r = base_r
    for n, m, amp in modes:
        r += base_r * amp * math.sin(n * theta) * math.cos(m * phi)
    return r


def spherical_to_cartesian(theta, phi, r):
    """Convert spherical (theta=polar, phi=azimuthal) to cartesian."""
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    return x, y, z


def rotate_x(x, y, z, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x, y * cos_a - z * sin_a, y * sin_a + z * cos_a


def rotate_z(x, y, z, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a, z


def project(x, y, z):
    """Orthographic projection to 2D. Returns (px, py, depth)."""
    return CENTER_X + x, CENTER_Y - y, z


def depth_style(z, z_min, z_max):
    """Compute stroke-width and opacity based on depth."""
    if z_max == z_min:
        t = 0.5
    else:
        t = (z - z_min) / (z_max - z_min)
    sw = STROKE_WIDTH_BACK + t * (STROKE_WIDTH_FRONT - STROKE_WIDTH_BACK)
    op = OPACITY_BACK + t * (OPACITY_FRONT - OPACITY_BACK)
    return sw, op


def circle_point(angle):
    """Point on the main circle at given angle (math convention)."""
    x = CENTER_X + BASE_RADIUS * math.cos(angle)
    y = CENTER_Y - BASE_RADIUS * math.sin(angle)
    return x, y


# =============================================================================
# Wireframe generation
# =============================================================================


def generate_latitude_line(theta, n_points, base_r, modes, tilt_x, tilt_z):
    points = []
    for i in range(n_points + 1):
        phi = 2 * math.pi * i / n_points
        r = distorted_radius(theta, phi, base_r, modes)
        x, y, z = spherical_to_cartesian(theta, phi, r)
        x, y, z = rotate_x(x, y, z, tilt_x)
        x, y, z = rotate_z(x, y, z, tilt_z)
        px, py, depth = project(x, y, z)
        points.append((px, py, depth))
    return points


def generate_longitude_line(phi, n_points, base_r, modes, tilt_x, tilt_z):
    points = []
    for i in range(n_points + 1):
        theta = math.pi * i / n_points
        r = distorted_radius(theta, phi, base_r, modes)
        x, y, z = spherical_to_cartesian(theta, phi, r)
        x, y, z = rotate_x(x, y, z, tilt_x)
        x, y, z = rotate_z(x, y, z, tilt_z)
        px, py, depth = project(x, y, z)
        points.append((px, py, depth))
    return points


def generate_sphere_lines(n_lat, n_lon, base_r, modes, tilt_x, tilt_z):
    lines = []
    for i in range(1, n_lat):
        theta = math.pi * i / n_lat
        points = generate_latitude_line(theta, POINTS_PER_LINE, base_r, modes, tilt_x, tilt_z)
        lines.append(points)
    for i in range(n_lon):
        phi = 2 * math.pi * i / n_lon
        points = generate_longitude_line(phi, POINTS_PER_LINE, base_r, modes, tilt_x, tilt_z)
        lines.append(points)
    return lines


def points_to_svg_path(points):
    if not points:
        return ""
    parts = [f"M {points[0][0]:.1f},{points[0][1]:.1f}"]
    for px, py, _ in points[1:]:
        parts.append(f"L {px:.1f},{py:.1f}")
    return " ".join(parts)


def average_depth(points):
    if not points:
        return 0
    return sum(p[2] for p in points) / len(points)


def compute_z_range(all_lines):
    z_min = float("inf")
    z_max = float("-inf")
    for points in all_lines:
        for _, _, z in points:
            z_min = min(z_min, z)
            z_max = max(z_max, z)
    return z_min, z_max


# =============================================================================
# Cutaway geometry
# =============================================================================


def build_cutaway_boundary():
    """Build the cubic bezier curve that defines the cutaway edge.

    Returns (ax, ay, bx, by, cp1x, cp1y, cp2x, cp2y) where:
    - A is the start point on the circle (upper-left area)
    - B is the end point on the circle (lower-right area)
    - CP1, CP2 are bezier control points
    """
    ax, ay = circle_point(CUTAWAY_ANGLE_A)
    bx, by = circle_point(CUTAWAY_ANGLE_B)

    cp1x = CENTER_X + BASE_RADIUS * CUTAWAY_CP1[0]
    cp1y = CENTER_Y - BASE_RADIUS * CUTAWAY_CP1[1]
    cp2x = CENTER_X + BASE_RADIUS * CUTAWAY_CP2[0]
    cp2y = CENTER_Y - BASE_RADIUS * CUTAWAY_CP2[1]

    return ax, ay, bx, by, cp1x, cp1y, cp2x, cp2y


def build_solid_overlay_path():
    """Build SVG path for the solid portion (circle minus cutaway).

    The path traces:
    1. Start at point A (on circle)
    2. Follow circle arc the "long way" (clockwise in SVG) to point B
    3. Bezier curve from B back to A (the cutaway boundary)
    4. Close
    """
    ax, ay, bx, by, cp1x, cp1y, cp2x, cp2y = build_cutaway_boundary()

    # Arc from A to B, going clockwise in SVG (sweep=1).
    # This is the "long way around" covering the solid portion.
    # Use large-arc-flag=1 since we want more than 180 degrees.
    d = (
        f"M {ax:.1f},{ay:.1f} "
        f"A {BASE_RADIUS} {BASE_RADIUS} 0 1 1 {bx:.1f},{by:.1f} "
        f"C {cp2x:.1f},{cp2y:.1f} {cp1x:.1f},{cp1y:.1f} {ax:.1f},{ay:.1f} "
        f"Z"
    )
    return d


def build_cutaway_edge_path():
    """Build SVG path for just the cutaway edge (bezier curve from A to B)."""
    ax, ay, bx, by, cp1x, cp1y, cp2x, cp2y = build_cutaway_boundary()
    d = f"M {ax:.1f},{ay:.1f} " f"C {cp1x:.1f},{cp1y:.1f} {cp2x:.1f},{cp2y:.1f} {bx:.1f},{by:.1f}"
    return d


# =============================================================================
# SVG output
# =============================================================================


def generate_svg(colors, output_path):
    lines = generate_sphere_lines(
        N_LATITUDE, N_LONGITUDE, BASE_RADIUS, DISTORTION_MODES, TILT_X, TILT_Z
    )
    z_min, z_max = compute_z_range(lines)

    # Sort wireframe lines by depth (back first)
    indexed = [(average_depth(pts), pts) for pts in lines]
    indexed.sort(key=lambda x: x[0])

    # Build SVG
    svg = []
    svg.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}">'
    )
    svg.append("  <!-- Surfaces Logo - Solid sphere with wireframe cutaway -->")
    svg.append("  <!-- Generated by generate_logo.py -->")

    # Defs: clip path and gradient
    svg.append("  <defs>")
    svg.append('    <clipPath id="sphere-clip">')
    svg.append(f'      <circle cx="{CENTER_X}" cy="{CENTER_Y}" r="{BASE_RADIUS}"/>')
    svg.append("    </clipPath>")
    svg.append(
        '    <radialGradient id="sphere-grad" ' 'cx="40%" cy="35%" r="65%" fx="40%" fy="35%">'
    )
    for offset, opacity in GRADIENT_STOPS:
        svg.append(
            f'      <stop offset="{offset:.0%}" '
            f'stop-color="{colors["gradient_base"]}" '
            f'stop-opacity="{opacity}"/>'
        )
    svg.append("    </radialGradient>")
    svg.append("  </defs>")

    # Layer 1: Wireframe mesh (clipped to circle, drawn underneath)
    svg.append('  <g id="wireframe" clip-path="url(#sphere-clip)">')
    for avg_z, points in indexed:
        sw, op = depth_style(avg_z, z_min, z_max)
        d = points_to_svg_path(points)
        svg.append(
            f'    <path d="{d}" '
            f'fill="none" stroke="{colors["wireframe"]}" '
            f'stroke-width="{sw:.2f}" '
            f'opacity="{op:.2f}" '
            f'stroke-linecap="round" '
            f'stroke-linejoin="round"/>'
        )
    svg.append("  </g>")

    # Layer 2: Solid overlay with cutaway (covers wireframe except in cutaway)
    solid_d = build_solid_overlay_path()
    svg.append(f'  <path id="solid-surface" d="{solid_d}" fill="url(#sphere-grad)"/>')

    # Layer 3: Cutaway edge line
    edge_d = build_cutaway_edge_path()
    svg.append(
        f'  <path id="cutaway-edge" d="{edge_d}" '
        f'fill="none" stroke="{colors["edge"]}" '
        f'stroke-width="{CUTAWAY_EDGE_WIDTH}" '
        f'opacity="0.6" '
        f'stroke-linecap="round"/>'
    )

    svg.append("</svg>")

    svg_content = "\n".join(svg)
    with open(output_path, "w") as f:
        f.write(svg_content)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Written: {output_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print("Generating Surfaces logos...")

    light_path = os.path.join(script_dir, "surfaces_logo.svg")
    generate_svg(COLORS_LIGHT, light_path)

    dark_path = os.path.join(script_dir, "surfaces_logo_dark.svg")
    generate_svg(COLORS_DARK, dark_path)

    print("Done.")
