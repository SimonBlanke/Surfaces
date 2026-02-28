#!/usr/bin/env python3
"""Generate plot compatibility matrix for the user guide.

Instantiates one representative function per category and checks which
plot methods are available via ``func.plot.available()``. Outputs an RST
file with the compatibility table.

Output Files
------------
docs/source/_generated/diagrams/
    plot_compatibility.rst

Usage
-----
    python -m docs._generators.generate_compatibility
"""

import sys
from typing import Dict, List, Tuple

from .config import DIAGRAMS_DIR, SRC_DIR

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# All plot types in display order
PLOT_TYPES = [
    "surface",
    "contour",
    "heatmap",
    "multi_slice",
    "fitness_distribution",
    "convergence",
    "latex",
]

# Display names for plot types (used as column headers)
PLOT_DISPLAY = {
    "surface": "Surface",
    "contour": "Contour",
    "heatmap": "Heatmap",
    "multi_slice": "Multi-Slice",
    "fitness_distribution": "Fitness Dist.",
    "convergence": "Convergence",
    "latex": "LaTeX/PDF",
}


def _get_representative_functions() -> List[Tuple[str, str, object]]:
    """Build list of (category_name, display_label, func_instance) tuples.

    Returns one representative, already-instantiated function per
    category/dimensionality combination.
    """
    representatives = []

    # --- Algebraic -----------------------------------------------------------
    from surfaces.test_functions.algebraic import (
        AckleyFunction,
        ForresterFunction,
        SphereFunction,
    )

    representatives.append(("algebraic_1d", "Algebraic 1D", ForresterFunction()))
    representatives.append(("algebraic_2d", "Algebraic 2D", AckleyFunction()))
    representatives.append(("algebraic_nd", "Algebraic N-D (n=5)", SphereFunction(n_dim=5)))

    # --- Constrained / Engineering -------------------------------------------
    from surfaces.test_functions.algebraic import (
        PressureVesselFunction,
        ThreeBarTrussFunction,
    )

    representatives.append(("engineering_2d", "Constrained 2D", ThreeBarTrussFunction()))
    representatives.append(("engineering_nd", "Constrained N-D (n=4)", PressureVesselFunction()))

    # --- BBOB ----------------------------------------------------------------
    try:
        from surfaces.test_functions.benchmark.bbob import Sphere as BBOBSphere

        representatives.append(("bbob_2d", "BBOB 2D", BBOBSphere(n_dim=2)))
        representatives.append(("bbob_nd", "BBOB N-D (n=5)", BBOBSphere(n_dim=5)))
    except ImportError:
        pass

    # --- CEC -----------------------------------------------------------------
    try:
        from surfaces.test_functions.benchmark.cec import cec2014

        if hasattr(cec2014, "__all__") and cec2014.__all__:
            cec_cls = getattr(cec2014, cec2014.__all__[0])
            representatives.append(("cec_2d", "CEC 2D", cec_cls(n_dim=2)))
            representatives.append(("cec_nd", "CEC N-D (n=10)", cec_cls(n_dim=10)))
    except (ImportError, Exception):
        pass

    # --- Simulation ----------------------------------------------------------
    try:
        from surfaces.test_functions.simulation import (
            DampedOscillatorFunction,
            LotkaVolterraFunction,
        )

        representatives.append(("simulation_2d", "Simulation 2D", DampedOscillatorFunction()))
        representatives.append(("simulation_nd", "Simulation N-D (n=4)", LotkaVolterraFunction()))
    except ImportError:
        pass

    # --- Machine Learning ----------------------------------------------------
    try:
        from surfaces.test_functions.machine_learning import (
            DecisionTreeClassifierFunction,
        )

        representatives.append(
            (
                "ml",
                "Machine Learning",
                DecisionTreeClassifierFunction(dataset="iris", cv=2),
            )
        )
    except ImportError:
        pass

    return representatives


def _build_matrix(
    representatives: List[Tuple[str, str, object]],
) -> List[Tuple[str, Dict[str, bool]]]:
    """Query func.plot.available() for each representative.

    Returns list of (display_label, {plot_name: bool}) tuples.
    """
    rows = []
    for _cat_id, label, func in representatives:
        available = set(func.plot.available())
        # convergence needs history; mark as conditional
        compat = {}
        for plot in PLOT_TYPES:
            if plot == "convergence":
                # Always possible if history is provided
                compat[plot] = True
            else:
                compat[plot] = plot in available
        rows.append((label, compat))
    return rows


def _render_rst_table(rows: List[Tuple[str, Dict[str, bool]]]) -> str:
    """Render the compatibility matrix as an RST list-table."""
    lines = []

    lines.append(".. list-table:: Plot Compatibility by Function Category")
    lines.append("   :header-rows: 1")
    lines.append("   :widths: 25 " + " ".join(["10"] * len(PLOT_TYPES)))
    lines.append("   :class: compatibility-matrix")
    lines.append("")

    # Header row
    lines.append("   * - Function Category")
    for plot in PLOT_TYPES:
        lines.append(f"     - {PLOT_DISPLAY[plot]}")

    # Data rows
    for label, compat in rows:
        lines.append(f"   * - {label}")
        for plot in PLOT_TYPES:
            if plot == "convergence":
                cell = "*"  # conditional (needs history)
            elif compat.get(plot, False):
                cell = "Y"
            else:
                cell = ""
            lines.append(f"     - {cell}")

    lines.append("")
    lines.append("| Y = available, * = requires evaluation history")
    lines.append("")

    return "\n".join(lines)


def generate() -> str:
    """Generate the full RST content for the compatibility section."""
    representatives = _get_representative_functions()
    rows = _build_matrix(representatives)
    return _render_rst_table(rows)


def main():
    """Generate and write the compatibility RST file."""
    DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DIAGRAMS_DIR / "plot_compatibility.rst"

    content = generate()
    output_path.write_text(content)
    print(f"  Generated: {output_path}")


if __name__ == "__main__":
    main()
