#!/usr/bin/env python3
"""Generate function catalog RST files.

This generator creates RST tables listing all test functions organized
by category. The generated files can be included in user guide pages.

Output Files
------------
docs/source/_generated/catalogs/
    algebraic_1d.rst
    algebraic_2d.rst
    algebraic_nd.rst
    ml_tabular_classification.rst
    ml_tabular_regression.rst
    ml_image_classification.rst
    ml_timeseries_classification.rst
    ml_timeseries_forecasting.rst
    engineering.rst
    summary.rst

Usage
-----
    python -m docs._generators.generate_catalogs
"""

from typing import List, Type

from . import extract_metadata, format_value, get_all_test_functions
from .config import CATALOGS_DIR, CATEGORY_DESCRIPTIONS, CATEGORY_DISPLAY_NAMES


def generate_catalog_table(
    functions: List[Type],
    title: str,
    description: str = "",
    ref_label: str = "",
) -> str:
    """
    Generate RST table for a list of functions.

    Parameters
    ----------
    functions : list
        List of test function classes.
    title : str
        Table title.
    description : str, optional
        Introductory text before the table.
    ref_label : str, optional
        RST reference label for cross-referencing.

    Returns
    -------
    str
        RST content.
    """
    lines = []

    # Add reference label if provided
    if ref_label:
        lines.extend([f".. _{ref_label}:", ""])

    # Add title
    lines.extend([title, "=" * len(title), ""])

    # Add description
    if description:
        lines.extend([description, ""])

    # If no functions, add a note
    if not functions:
        lines.extend(["*No functions available in this category.*", ""])
        return "\n".join(lines)

    # Create the table
    lines.extend(
        [
            ".. list-table::",
            "   :widths: 30 10 12 48",
            "   :header-rows: 1",
            "   :class: function-catalog-table",
            "",
            "   * - Function",
            "     - Dims",
            "     - f(x*)",
            "     - Description",
        ]
    )

    # Sort functions by name
    for func_class in sorted(functions, key=lambda f: f.__name__):
        meta = extract_metadata(func_class)

        # Format the class reference
        class_ref = f":class:`~{meta['module']}.{meta['name']}`"

        # Get first line of description, truncate if needed
        desc = meta["first_line"]
        if len(desc) > 60:
            desc = desc[:57] + "..."

        lines.extend(
            [
                f"   * - {class_ref}",
                f"     - {format_value(meta['n_dim'])}",
                f"     - {format_value(meta['f_global'])}",
                f"     - {desc}",
            ]
        )

    return "\n".join(lines)


def generate_summary_table(categories: dict) -> str:
    """
    Generate summary table of all categories.

    Parameters
    ----------
    categories : dict
        Dictionary of category names to function lists.

    Returns
    -------
    str
        RST content with summary table.
    """
    lines = [
        ".. _test_functions_overview:",
        "",
        "Test Functions Overview",
        "=======================",
        "",
        "Summary of all available test function categories.",
        "",
        ".. list-table::",
        "   :widths: 25 10 65",
        "   :header-rows: 1",
        "",
        "   * - Category",
        "     - Count",
        "     - Description",
    ]

    total = 0
    for cat_key, funcs in categories.items():
        if not funcs:
            continue

        count = len(funcs)
        total += count

        display_name = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
        description = CATEGORY_DESCRIPTIONS.get(cat_key, "")

        lines.extend(
            [
                f"   * - {display_name}",
                f"     - {count}",
                f"     - {description}",
            ]
        )

    # Add total row
    lines.extend(
        [
            "   * - **Total**",
            f"     - **{total}**",
            "     - ",
        ]
    )

    return "\n".join(lines)


def main():
    """Generate all catalog files."""
    categories = get_all_test_functions()

    print(f"Output directory: {CATALOGS_DIR}")

    # Generate individual category files
    for cat_key, funcs in categories.items():
        title = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
        description = CATEGORY_DESCRIPTIONS.get(cat_key, "")
        ref_label = f"catalog_{cat_key}"

        content = generate_catalog_table(funcs, title, description, ref_label)

        output_path = CATALOGS_DIR / f"{cat_key}.rst"
        output_path.write_text(content)
        print(f"  Generated: {output_path.name} ({len(funcs)} functions)")

    # Generate summary file
    summary_content = generate_summary_table(categories)
    summary_path = CATALOGS_DIR / "summary.rst"
    summary_path.write_text(summary_content)

    total = sum(len(f) for f in categories.values())
    print(f"  Generated: summary.rst ({total} total functions)")


if __name__ == "__main__":
    main()
