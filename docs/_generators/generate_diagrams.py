#!/usr/bin/env python3
"""Generate architecture diagrams from code structure.

This generator creates Mermaid diagrams and RST tables showing
the module hierarchy, class inheritance, and feature matrix.

Output Files
------------
docs/source/_generated/diagrams/
    module_hierarchy.md    - Mermaid flowchart of modules
    class_hierarchy.md     - Mermaid class diagram
    feature_matrix.rst     - RST table of function capabilities

Usage
-----
    python -m docs._generators.generate_diagrams
"""

from pathlib import Path
from typing import Dict, List, Type

from . import count_by_category, get_all_test_functions
from .config import CATEGORY_DISPLAY_NAMES, DIAGRAMS_DIR


def generate_module_hierarchy() -> str:
    """
    Generate Mermaid diagram of module structure.

    Returns
    -------
    str
        Markdown content with Mermaid diagram.
    """
    return """# Module Hierarchy

The following diagram shows the organization of test functions in Surfaces.

```mermaid
flowchart TD
    subgraph surfaces.test_functions
        A[test_functions] --> B[algebraic]
        A --> C[machine_learning]
        A --> D[engineering]
        A --> E[bbob]
        A --> F[cec]

        subgraph algebraic
            B --> B1[test_functions_1d]
            B --> B2[test_functions_2d]
            B --> B3[test_functions_nd]
        end

        subgraph machine_learning
            C --> C1[tabular]
            C --> C2[image]
            C --> C3[timeseries]
            C1 --> C1a[classification]
            C1 --> C1b[regression]
            C2 --> C2a[classification]
            C3 --> C3a[classification]
            C3 --> C3b[forecasting]
        end
    end
```
"""


def generate_class_hierarchy() -> str:
    """
    Generate Mermaid diagram of class inheritance.

    Returns
    -------
    str
        Markdown content with Mermaid class diagram.
    """
    counts = count_by_category()

    n_algebraic = counts.get("algebraic_1d", 0) + counts.get("algebraic_2d", 0) + counts.get("algebraic_nd", 0)
    n_ml = sum(v for k, v in counts.items() if k.startswith("ml_"))
    n_eng = counts.get("engineering", 0)

    return f"""# Class Hierarchy

Inheritance structure of test function classes.

```mermaid
classDiagram
    BaseTestFunction <|-- AlgebraicFunction
    BaseTestFunction <|-- MLTestFunction
    BaseTestFunction <|-- EngineeringFunction

    AlgebraicFunction <|-- SphereFunction
    AlgebraicFunction <|-- AckleyFunction
    AlgebraicFunction <|-- RosenbrockFunction
    note for AlgebraicFunction "{n_algebraic} concrete classes"

    MLTestFunction <|-- KNeighborsClassifierFunction
    MLTestFunction <|-- GradientBoostingRegressorFunction
    note for MLTestFunction "{n_ml} concrete classes"

    EngineeringFunction <|-- WeldedBeamFunction
    EngineeringFunction <|-- PressureVesselFunction
    note for EngineeringFunction "{n_eng} concrete classes"

    class BaseTestFunction {{
        +__call__(params)
        +search_space()
        +to_scipy()
        +objective: str
        +sleep: float
        +memory: bool
        +callbacks: list
        +noise: BaseNoise
    }}
```
"""


def generate_feature_matrix() -> str:
    """
    Generate RST table showing function capabilities by category.

    Returns
    -------
    str
        RST content with feature matrix table.
    """
    categories = get_all_test_functions()

    lines = [
        ".. _feature_matrix:",
        "",
        "Feature Matrix",
        "==============",
        "",
        "Capabilities of each test function category.",
        "",
        ".. list-table::",
        "   :widths: 22 8 8 8 10 10 10 10",
        "   :header-rows: 1",
        "   :class: feature-matrix-table",
        "",
        "   * - Category",
        "     - Count",
        "     - Scalable",
        "     - LaTeX",
        "     - Constraints",
        "     - Global Known",
        "     - Noise",
        "     - scipy",
    ]

    # Define features for each category
    category_features = {
        "algebraic_1d": {
            "scalable": "No",
            "latex": "Yes",
            "constraints": "No",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
        "algebraic_2d": {
            "scalable": "No",
            "latex": "Yes",
            "constraints": "No",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
        "algebraic_nd": {
            "scalable": "Yes",
            "latex": "Yes",
            "constraints": "No",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
        "ml_tabular_classification": {
            "scalable": "No",
            "latex": "No",
            "constraints": "No",
            "global": "No",
            "noise": "Yes",
            "scipy": "No",
        },
        "ml_tabular_regression": {
            "scalable": "No",
            "latex": "No",
            "constraints": "No",
            "global": "No",
            "noise": "Yes",
            "scipy": "No",
        },
        "ml_image_classification": {
            "scalable": "No",
            "latex": "No",
            "constraints": "No",
            "global": "No",
            "noise": "Yes",
            "scipy": "No",
        },
        "ml_timeseries_classification": {
            "scalable": "No",
            "latex": "No",
            "constraints": "No",
            "global": "No",
            "noise": "Yes",
            "scipy": "No",
        },
        "ml_timeseries_forecasting": {
            "scalable": "No",
            "latex": "No",
            "constraints": "No",
            "global": "No",
            "noise": "Yes",
            "scipy": "No",
        },
        "engineering": {
            "scalable": "No",
            "latex": "No",
            "constraints": "Yes",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
        "bbob": {
            "scalable": "Yes",
            "latex": "No",
            "constraints": "No",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
        "cec": {
            "scalable": "Yes",
            "latex": "No",
            "constraints": "No",
            "global": "Yes",
            "noise": "Yes",
            "scipy": "Yes",
        },
    }

    for cat_key, funcs in categories.items():
        if not funcs:
            continue

        feat = category_features.get(cat_key, {})
        if not feat:
            continue

        name = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)

        lines.extend(
            [
                f"   * - {name}",
                f"     - {len(funcs)}",
                f"     - {feat.get('scalable', '---')}",
                f"     - {feat.get('latex', '---')}",
                f"     - {feat.get('constraints', '---')}",
                f"     - {feat.get('global', '---')}",
                f"     - {feat.get('noise', '---')}",
                f"     - {feat.get('scipy', '---')}",
            ]
        )

    return "\n".join(lines)


def main():
    """Generate all diagram files."""
    print(f"Output directory: {DIAGRAMS_DIR}")

    # Module hierarchy
    hierarchy = generate_module_hierarchy()
    output_path = DIAGRAMS_DIR / "module_hierarchy.md"
    output_path.write_text(hierarchy)
    print(f"  Generated: {output_path.name}")

    # Class hierarchy
    classes = generate_class_hierarchy()
    output_path = DIAGRAMS_DIR / "class_hierarchy.md"
    output_path.write_text(classes)
    print(f"  Generated: {output_path.name}")

    # Feature matrix
    matrix = generate_feature_matrix()
    output_path = DIAGRAMS_DIR / "feature_matrix.rst"
    output_path.write_text(matrix)
    print(f"  Generated: {output_path.name}")


if __name__ == "__main__":
    main()
