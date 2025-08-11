#!/usr/bin/env python3
"""
Generate documentation tables showing parameter ranges covered by the database
for each machine learning test function.

This script analyzes the stored search data and creates markdown tables
documenting the parameter coverage for each ML function.
"""

import sys
import os
import sqlite3
import json
from typing import Dict, List, Any, Set, Union
from collections import defaultdict

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from surfaces.test_functions.machine_learning.tabular.regression.test_functions.gradient_boosting_regressor import (
    GradientBoostingRegressorFunction,
)
from surfaces.test_functions.machine_learning.tabular.regression.test_functions.k_neighbors_regressor import (
    KNeighborsRegressorFunction,
)
from surfaces.test_functions.machine_learning.tabular.classification.test_functions.k_neighbors_classifier import (
    KNeighborsClassifierFunction,
)

# Available ML functions
ML_FUNCTIONS = {
    "gradient_boosting_regressor": GradientBoostingRegressorFunction,
    "k_neighbors_regressor": KNeighborsRegressorFunction,
    "k_neighbors_classifier": KNeighborsClassifierFunction,
}


def get_parameter_ranges_from_db(
    function_name: str, data_dir: str = None
) -> Dict[str, Any]:
    """
    Extract parameter ranges from stored database data.

    Args:
        function_name: Name of the ML function
        data_dir: Directory containing search data databases

    Returns:
        Dictionary containing parameter statistics and ranges
    """
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "src", "surfaces", "search_data"
        )

    db_path = os.path.join(data_dir, f"{function_name}.db")

    if not os.path.exists(db_path):
        return {
            "exists": False,
            "message": "No database found - run data collection first",
        }

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Get all parameter data
            cursor = conn.execute("SELECT * FROM search_data")
            rows = cursor.fetchall()

            if not rows:
                return {
                    "exists": True,
                    "message": "Database exists but contains no data",
                }

            # Analyze parameter ranges
            param_stats = defaultdict(
                lambda: {
                    "type": None,
                    "values": set(),
                    "numeric_values": [],
                    "min": None,
                    "max": None,
                    "unique_count": 0,
                }
            )

            # Exclude non-parameter columns
            non_param_cols = {"id", "score", "evaluation_time", "timestamp"}

            for row in rows:
                for col_name in row.keys():
                    if col_name in non_param_cols:
                        continue

                    value = row[col_name]
                    param_stats[col_name]["values"].add(value)

                    # Try to determine if it's numeric
                    try:
                        numeric_val = float(value)
                        param_stats[col_name]["numeric_values"].append(numeric_val)
                        if param_stats[col_name]["type"] is None:
                            param_stats[col_name]["type"] = "numeric"
                    except ValueError:
                        if param_stats[col_name]["type"] != "numeric":
                            param_stats[col_name]["type"] = "categorical"

            # Finalize statistics
            result = {"exists": True, "total_evaluations": len(rows), "parameters": {}}

            for param_name, stats in param_stats.items():
                unique_values = list(stats["values"])
                param_info = {
                    "type": stats["type"],
                    "unique_count": len(unique_values),
                    "unique_values": sorted(unique_values),
                }

                if stats["type"] == "numeric" and stats["numeric_values"]:
                    param_info["min"] = min(stats["numeric_values"])
                    param_info["max"] = max(stats["numeric_values"])
                    param_info["range"] = f"{param_info['min']} - {param_info['max']}"

                result["parameters"][param_name] = param_info

            return result

    except Exception as e:
        return {"exists": True, "error": str(e)}


def get_default_parameter_ranges(function_class) -> Dict[str, Any]:
    """
    Get the default parameter ranges defined in the function class.

    Args:
        function_class: ML function class

    Returns:
        Dictionary with default parameter ranges
    """
    func = function_class()
    search_space = func.search_space()

    result = {}
    for param_name, param_values in search_space.items():
        param_info = {
            "type": (
                "categorical"
                if isinstance(param_values, list)
                and any(isinstance(v, str) for v in param_values)
                else "numeric"
            ),
            "default_count": len(param_values),
            "default_values": (
                param_values[:5] if len(param_values) > 5 else param_values
            ),  # Show first 5
            "is_truncated": len(param_values) > 5,
        }

        # For numeric ranges
        if param_info["type"] == "numeric" and all(
            isinstance(v, (int, float)) for v in param_values
        ):
            param_info["default_min"] = min(param_values)
            param_info["default_max"] = max(param_values)
            param_info["default_range"] = (
                f"{param_info['default_min']} - {param_info['default_max']}"
            )

        result[param_name] = param_info

    return result


def generate_markdown_table(function_name: str, function_class) -> str:
    """
    Generate a markdown table for a single ML function showing parameter coverage.

    Args:
        function_name: Name of the ML function
        function_class: ML function class

    Returns:
        Markdown table as string
    """
    # Get both default and stored data
    default_ranges = get_default_parameter_ranges(function_class)
    stored_ranges = get_parameter_ranges_from_db(function_name)

    # Create function instance for metadata
    func = function_class()
    display_name = getattr(func, "name", function_name.replace("_", " ").title())

    markdown = f"## {display_name}\n\n"
    markdown += f"**Function ID:** `{function_name}`\n\n"

    if not stored_ranges.get("exists", False):
        markdown += "⚠️ **No database found** - Run data collection first using:\n"
        markdown += (
            f"```bash\npython collect_ml_search_data.py {function_name}\n```\n\n"
        )

        # Show default ranges
        markdown += "### Default Parameter Ranges\n\n"
        markdown += "| Parameter | Type | Count | Range/Values | Notes |\n"
        markdown += "|-----------|------|-------|--------------|-------|\n"

        for param_name, info in default_ranges.items():
            if info["type"] == "numeric":
                range_str = info.get("default_range", "N/A")
            else:
                values_str = str(info["default_values"])
                if info.get("is_truncated", False):
                    values_str += f" (showing first 5 of {info['default_count']})"
                range_str = values_str

            notes = "Default values defined in code"
            markdown += f"| {param_name} | {info['type']} | {info['default_count']} | {range_str} | {notes} |\n"

        return markdown + "\n"

    if "error" in stored_ranges:
        markdown += f"❌ **Database error:** {stored_ranges['error']}\n\n"
        return markdown

    if "message" in stored_ranges:
        markdown += f"ℹ️ **Status:** {stored_ranges['message']}\n\n"
        return markdown

    markdown += (
        f"**Total evaluations stored:** {stored_ranges['total_evaluations']:,}\n\n"
    )

    # Compare default vs stored
    markdown += "### Parameter Coverage\n\n"
    markdown += "| Parameter | Type | Stored Values | Coverage Range | Default Range | Coverage |\n"
    markdown += "|-----------|------|---------------|----------------|---------------|----------|\n"

    for param_name in default_ranges.keys():
        default_info = default_ranges[param_name]
        stored_info = stored_ranges.get("parameters", {}).get(param_name)

        if stored_info is None:
            coverage = "❌ Not found in database"
            stored_count = "0"
            stored_range = "N/A"
        else:
            stored_count = str(stored_info["unique_count"])

            if stored_info["type"] == "numeric":
                stored_range = stored_info.get("range", "N/A")
            else:
                if stored_info["unique_count"] <= 3:
                    stored_range = str(stored_info["unique_values"])
                else:
                    stored_range = f"{stored_info['unique_count']} unique values"

            # Calculate coverage percentage
            coverage_pct = (
                stored_info["unique_count"] / default_info["default_count"]
            ) * 100
            if coverage_pct >= 100:
                coverage = "✅ Complete"
            elif coverage_pct >= 80:
                coverage = f"🟡 {coverage_pct:.1f}%"
            elif coverage_pct >= 50:
                coverage = f"🟠 {coverage_pct:.1f}%"
            else:
                coverage = f"🔴 {coverage_pct:.1f}%"

        # Default range
        if default_info["type"] == "numeric":
            default_range = default_info.get("default_range", "N/A")
        else:
            default_range = f"{default_info['default_count']} categories"

        markdown += f"| {param_name} | {default_info['type']} | {stored_count} | {stored_range} | {default_range} | {coverage} |\n"

    return markdown + "\n"


def generate_all_docs(output_file: str = "ML_PARAMETER_COVERAGE.md") -> None:
    """
    Generate complete documentation for all ML functions.

    Args:
        output_file: Path to output markdown file
    """
    markdown_content = """# Machine Learning Test Functions - Parameter Coverage

    This document shows the parameter ranges covered by the database for each machine learning test function in the Surfaces library.

    The database stores evaluation results for specific parameter combinations to enable fast lookup of expensive ML computations. Each function has default parameter ranges defined in code, and this document shows how much of that space is covered by stored data.

    ## Legend
    - ✅ **Complete**: All default parameter combinations are stored
    - 🟡 **Good coverage**: 80-99% of default combinations stored  
    - 🟠 **Partial coverage**: 50-79% of default combinations stored
    - 🔴 **Limited coverage**: <50% of default combinations stored
    - ❌ **Not found**: Parameter not found in stored data

    ## Usage

    To collect data for any function, use:
    ```bash
    # Collect for all functions
    python collect_ml_search_data.py --all

    # Collect for specific function
    python collect_ml_search_data.py <function_name>

    # Check current status
    python collect_ml_search_data.py --list
    ```

    ---

    """

    for function_name, function_class in ML_FUNCTIONS.items():
        print(f"Generating documentation for {function_name}...")
        markdown_content += generate_markdown_table(function_name, function_class)

    # Add generation timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    markdown_content += f"\n---\n\n*Generated on {timestamp}*\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(markdown_content)

    print(f"Documentation generated: {output_file}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate parameter coverage documentation for ML test functions"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="ML_PARAMETER_COVERAGE.md",
        help="Output markdown file (default: ML_PARAMETER_COVERAGE.md)",
    )

    parser.add_argument(
        "--function",
        "-f",
        choices=list(ML_FUNCTIONS.keys()),
        help="Generate docs for specific function only",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw data as JSON instead of markdown",
    )

    args = parser.parse_args()

    if args.json:
        # Output raw JSON data
        all_data = {}
        for func_name, func_class in ML_FUNCTIONS.items():
            if args.function and func_name != args.function:
                continue
            all_data[func_name] = {
                "default_ranges": get_default_parameter_ranges(func_class),
                "stored_ranges": get_parameter_ranges_from_db(func_name),
            }

        output = (
            args.output
            if args.output.endswith(".json")
            else args.output.replace(".md", ".json")
        )
        with open(output, "w") as f:
            json.dump(all_data, f, indent=2, default=str)
        print(f"JSON data written to: {output}")

    elif args.function:
        # Generate docs for single function
        func_class = ML_FUNCTIONS[args.function]
        markdown = generate_markdown_table(args.function, func_class)
        print(markdown)

    else:
        # Generate complete documentation
        generate_all_docs(args.output)


if __name__ == "__main__":
    main()
