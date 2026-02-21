# Logo Creation Prompt for Surfaces Project

## Task

Create an SVG logo for the **Surfaces** Python library. The logo should look like a **stylized 3D surface plot** (think matplotlib/MATLAB surface plots showing optimization landscapes with peaks and valleys), but simplified and bold enough to work as a logo.

## Project Context

- **Surfaces** is a Python library providing mathematical test functions (Rastrigin, Ackley, Rosenbrock, etc.) for benchmarking optimization algorithms
- It is part of the "GFO Stack" alongside Gradient-Free-Optimizers and Hyperactive
- The existing drop_wave surface visualization is at `docs/source/_static/drop_wave_surface.svg` for reference on what these surfaces look like
- The README expects logo files at:
  - `docs/source/_static/surfaces_logo.svg` (light mode, dark lines on transparent)
  - `docs/source/_static/surfaces_logo_dark.svg` (dark mode, light lines on transparent)

## Design Requirements

### Visual Concept
- A **3D surface plot** viewed from a perspective/isometric angle
- Shows a mathematical surface with visible **peaks and valleys** (like an optimization landscape)
- **Stylized for logo use**: clean lines, not a literal plot. No axes, no labels, no grid floor.
- Should have a clear bold silhouette that reads well at small sizes (favicons, badges)
- Detail rewards closer inspection (the surface mesh lines, the mathematical curvature)

### Sibling Logo Style Reference
The user creates logos with mathematical patterns. The two sibling logos use:
- **Hyperactive** (`/home/me/github-workspace/001-hyperactive-project/hyperactive_logo.svg`): A single spiral path reused ~30 times with progressive rotation+scaling matrix transforms, creating a fractal spiral. Deep blue (#0008e6, #323161). Bold shape at a glance (spirals), complex on inspection.
- **GFO** (`/home/me/github-workspace/002-gfo-stack/Gradient-Free-Optimizers/docs/images/gradient_logo_ink.svg`): Concentric arc segments at different radii/rotations with an ellipse. Deep blue (#0008e6, #000054) and light blue (#80b3ff). Bold shape at a glance (circles), layered detail on inspection.

Key takeaway: **simple bold shape at a distance, mathematical detail up close**.

### Color Palette
Use **deep blue** to match the sibling projects:
- Primary: `#0008e6`
- Dark accent: `#323161` / `#000054`
- Light accent: `#80b3ff` / `#80b4ff`
- Dark mode variant uses the lighter blues on transparent background

### Technical Approach
- Write a **Python generator script** at `docs/source/_static/generate_logo.py` that outputs SVG
- Use mathematical functions to compute the surface (e.g., a Rastrigin-like or custom function with interesting peaks/valleys)
- Project the 3D surface to 2D using perspective or isometric projection
- Render the surface as mesh lines (latitude/longitude style grid on the surface) with depth-based opacity/stroke-width
- The surface itself should be the bold shape. The mesh lines are the detail.
- Consider using filled polygons between mesh lines (with opacity/color variation) to give the surface visual weight and silhouette, not just wireframe lines
- Output clean SVG with viewBox, no Inkscape-specific attributes
- Generate both light and dark variants

### What Did NOT Work (Previous Attempts)
1. **Pure wireframe mesh sphere**: Too complex, no bold primary shape. Just thin lines with no structure visible at a distance.
2. **Solid sphere with wireframe cutaway**: The cutaway concept did not look good in practice.

The key lesson: the logo needs a **recognizable bold shape** as its foundation. A 3D surface plot naturally provides this since the surface silhouette (the mountain/valley profile) is itself a bold shape.

### Suggestions
- Consider a surface like `z = sin(x) * cos(y)` or a simplified Rastrigin with 2-3 peaks/valleys
- The viewing angle should show the surface character well (not too flat, not too steep)
- Filled mesh quads (with depth shading) would give the surface visual mass
- The surface outline/silhouette should be clean and distinctive
- Keep the surface compact (roughly square or circular bounding shape)
- Total line count in SVG should be reasonable (aim for <50KB)

## Files

| File | Purpose |
|------|---------|
| `docs/source/_static/generate_logo.py` | Generator script (overwrite existing) |
| `docs/source/_static/surfaces_logo.svg` | Light mode output |
| `docs/source/_static/surfaces_logo_dark.svg` | Dark mode output |
| `docs/source/_static/drop_wave_surface.svg` | Reference: what a real surface plot looks like |
