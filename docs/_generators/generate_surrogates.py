#!/usr/bin/env python3
"""Generate surrogate model coverage and metrics documentation.

This generator creates RST tables showing which ML functions have
pre-trained surrogate models, along with their accuracy and speedup metrics.

Output Files
------------
docs/source/_generated/surrogates/
    coverage.rst          - Coverage table with accuracy/speedup metrics

Usage
-----
    python -m docs._generators.generate_surrogates

    # With live validation (slow, recalculates metrics)
    python -m docs._generators.generate_surrogates --validate

    # Specify number of validation samples
    python -m docs._generators.generate_surrogates --validate --samples 50
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from .config import SURROGATES_DIR


def get_surrogate_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all registered ML functions and their surrogates.

    Returns
    -------
    dict
        Function name -> {
            "exists": bool,
            "path": Path or None,
            "metadata": dict or None,
            "class_name": str,
            "category": str,  # "classification" or "regression"
        }
    """
    try:
        from surfaces._surrogates._ml_registry import (
            get_function_config,
            get_registered_functions,
        )
        from surfaces._surrogates._onnx_utils import (
            get_metadata_path,
            get_surrogate_model_path,
        )
    except ImportError as e:
        print(f"Warning: Could not import surrogate modules: {e}")
        return {}

    result = {}
    for name in get_registered_functions():
        config = get_function_config(name)
        model_path = get_surrogate_model_path(name)
        meta_path = get_metadata_path(name)

        # Load metadata if available
        metadata = None
        if meta_path is not None:
            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass

        # Determine category
        category = "classification" if "classifier" in name else "regression"

        result[name] = {
            "exists": model_path is not None,
            "path": model_path,
            "metadata": metadata,
            "class_name": config["class"].__name__,
            "category": category,
        }

    return result


def validate_surrogate(
    function_name: str,
    n_samples: int = 50,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run validation for a surrogate model.

    Parameters
    ----------
    function_name : str
        Name of the registered function.
    n_samples : int
        Number of random samples for validation.
    verbose : bool
        Print progress.

    Returns
    -------
    dict or None
        Validation results with metrics and timing, or None if validation fails.
    """
    try:
        from surfaces._surrogates._ml_registry import get_function_config
        from surfaces._surrogates._surrogate_validator import SurrogateValidator
    except ImportError:
        return None

    try:
        config = get_function_config(function_name)
        FuncClass = config["class"]

        # Create function instance with default fixed params
        fixed_params = config["fixed_params"]
        init_kwargs = {}

        # Use first value from each fixed param
        for key, values in fixed_params.items():
            init_kwargs[key] = values[0]

        func = FuncClass(**init_kwargs, use_surrogate=False)
        validator = SurrogateValidator(func)

        results = validator.validate_random(n_samples=n_samples, verbose=verbose)
        return results

    except Exception as e:
        if verbose:
            print(f"  Validation failed for {function_name}: {e}")
        return None


def generate_coverage_table(
    status: Dict[str, Dict[str, Any]],
    validation_results: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
    """Generate RST table showing surrogate coverage and metrics.

    Parameters
    ----------
    status : dict
        Output from get_surrogate_status().
    validation_results : dict, optional
        Function name -> validation results from validate_surrogate().

    Returns
    -------
    str
        RST content.
    """
    validation_results = validation_results or {}

    lines = [
        ".. _surrogate_coverage:",
        "",
        "Surrogate Model Coverage",
        "========================",
        "",
        "The following table shows which ML functions have pre-trained surrogate",
        "models available.",
        "",
    ]

    # Separate by category
    classifiers = {k: v for k, v in status.items() if v["category"] == "classification"}
    regressors = {k: v for k, v in status.items() if v["category"] == "regression"}

    # Classification table
    lines.extend(
        _generate_category_table(
            "Classification Functions",
            classifiers,
            validation_results,
        )
    )

    lines.append("")

    # Regression table
    lines.extend(
        _generate_category_table(
            "Regression Functions",
            regressors,
            validation_results,
        )
    )

    # Add legend
    lines.extend(
        [
            "",
            "**Legend:**",
            "",
            "- **Status**: Available = pre-trained model exists, Missing = not yet trained",
            "- **R²**: Coefficient of determination (higher is better, 1.0 = perfect)",
            "- **Speedup**: How many times faster than real evaluation",
            "- **Samples**: Number of training samples used",
            "",
            "Metrics are from model metadata or live validation. Missing values indicate",
            "the metric was not available.",
        ]
    )

    return "\n".join(lines)


def _generate_category_table(
    title: str,
    functions: Dict[str, Dict[str, Any]],
    validation_results: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Generate RST table for a category of functions."""
    lines = [
        title,
        "-" * len(title),
        "",
        ".. list-table::",
        "   :widths: 28 12 12 12 12 24",
        "   :header-rows: 1",
        "   :class: surrogate-coverage-table",
        "",
        "   * - Function",
        "     - Status",
        "     - R²",
        "     - Speedup",
        "     - Samples",
        "     - Notes",
    ]

    for name in sorted(functions.keys()):
        info = functions[name]
        validation = validation_results.get(name, {})

        # Status
        if info["exists"]:
            status = "Available"
            status_class = ""
        else:
            status = "Missing"
            status_class = ""

        # Get metrics from metadata or validation
        r2 = "---"
        speedup = "---"
        n_samples = "---"
        notes = ""

        # Get R² and samples from metadata (more reliable, from full training)
        if info["metadata"]:
            meta = info["metadata"]
            if "training_r2" in meta:
                r2 = f"{meta['training_r2']:.3f}"
            if "n_samples" in meta:
                n_samples = str(meta["n_samples"])

        # Get speedup from live validation (not stored in metadata)
        if validation and "timing" in validation:
            timing = validation["timing"]
            if "speedup" in timing and timing["speedup"] > 0:
                speedup = f"{timing['speedup']:.0f}x"

        # Generate notes
        if info["exists"]:
            if info["metadata"]:
                meta = info["metadata"]
                if meta.get("has_validity_model"):
                    notes = "Has validity model"
        else:
            notes = "Training pending"

        # Class reference
        class_ref = f":class:`~surfaces.test_functions.machine_learning.{info['class_name']}`"

        lines.extend(
            [
                f"   * - {class_ref}",
                f"     - {status}",
                f"     - {r2}",
                f"     - {speedup}",
                f"     - {n_samples}",
                f"     - {notes}",
            ]
        )

    return lines


def generate_summary_stats(status: Dict[str, Dict[str, Any]]) -> str:
    """Generate summary statistics RST content."""
    total = len(status)
    available = sum(1 for s in status.values() if s["exists"])
    missing = total - available

    coverage_pct = (available / total * 100) if total > 0 else 0

    lines = [
        "",
        "Summary",
        "-------",
        "",
        f"- **Total registered functions**: {total}",
        f"- **Surrogates available**: {available}",
        f"- **Surrogates missing**: {missing}",
        f"- **Coverage**: {coverage_pct:.0f}%",
    ]

    return "\n".join(lines)


def main():
    """Generate all surrogate documentation files."""
    parser = argparse.ArgumentParser(
        description="Generate surrogate model coverage documentation",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run live validation to calculate accuracy/speedup (slow)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples for validation (default: 50)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    SURROGATES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {SURROGATES_DIR}")

    # Get surrogate status
    print("Checking surrogate status...")
    status = get_surrogate_status()

    if not status:
        print("  Warning: No ML functions found (sklearn may not be installed)")
        # Generate placeholder content
        content = _generate_placeholder_content()
        output_path = SURROGATES_DIR / "coverage.rst"
        output_path.write_text(content)
        print(f"  Generated: {output_path.name} (placeholder)")
        return

    available = sum(1 for s in status.values() if s["exists"])
    print(f"  Found {len(status)} registered functions, {available} with surrogates")

    # Run validation if requested
    validation_results = {}
    if args.validate:
        print(f"Running validation with {args.samples} samples per function...")
        for name, info in status.items():
            if info["exists"]:
                print(f"  Validating {name}...")
                results = validate_surrogate(
                    name,
                    n_samples=args.samples,
                    verbose=args.verbose,
                )
                if results:
                    validation_results[name] = results
                    r2 = results["metrics"]["r2"]
                    speedup = results["timing"]["speedup"]
                    print(f"    R²={r2:.3f}, Speedup={speedup:.0f}x")

    # Generate coverage table
    content = generate_coverage_table(status, validation_results)
    content += "\n" + generate_summary_stats(status)

    output_path = SURROGATES_DIR / "coverage.rst"
    output_path.write_text(content)
    print(f"  Generated: {output_path.name}")


def _generate_placeholder_content() -> str:
    """Generate placeholder content when sklearn is not available."""
    return """.. _surrogate_coverage:

Surrogate Model Coverage
========================

.. note::

   Surrogate coverage information is not available because scikit-learn
   is not installed. Install it with:

   .. code-block:: bash

       pip install surfaces[ml]

   Then rebuild the documentation.
"""


if __name__ == "__main__":
    main()
