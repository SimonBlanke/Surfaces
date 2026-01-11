#!/usr/bin/env python3
"""Master script to run all documentation generators.

This script is the entry point for generating all documentation assets.
It can be run directly or via `make docs-generate`.

Usage
-----
Run all generators:
    python -m docs._generators.generate_all

Run specific generators:
    python -m docs._generators.generate_all --catalogs
    python -m docs._generators.generate_all --plots
    python -m docs._generators.generate_all --diagrams

Clean and regenerate:
    python -m docs._generators.generate_all --clean --all
"""

import argparse
import shutil
import sys
from pathlib import Path

# Ensure we can import from the generators package
GENERATORS_DIR = Path(__file__).parent
PROJECT_ROOT = GENERATORS_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from docs._generators.config import GENERATED_DIR, ensure_directories  # noqa: E402


def clean_generated():
    """Remove all generated files."""
    if GENERATED_DIR.exists():
        print(f"Cleaning {GENERATED_DIR}")
        shutil.rmtree(GENERATED_DIR)
    ensure_directories()


def run_catalog_generator():
    """Run the catalog table generator."""
    print("\n" + "=" * 60)
    print("Generating Function Catalogs")
    print("=" * 60)

    try:
        from docs._generators import generate_catalogs

        generate_catalogs.main()
    except ImportError as e:
        print(f"Warning: Could not import catalog generator: {e}")
    except Exception as e:
        print(f"Error generating catalogs: {e}")
        raise


def run_plot_generator():
    """Run the plot/visualization generator."""
    print("\n" + "=" * 60)
    print("Generating Plots")
    print("=" * 60)

    try:
        from docs._generators import generate_plots

        generate_plots.main()
    except ImportError as e:
        print(f"Warning: Could not import plot generator: {e}")
        print("  (This may be expected if plotly/kaleido are not installed)")
    except Exception as e:
        print(f"Error generating plots: {e}")
        # Don't raise - plots are optional


def run_diagram_generator():
    """Run the architecture diagram generator."""
    print("\n" + "=" * 60)
    print("Generating Diagrams")
    print("=" * 60)

    try:
        from docs._generators import generate_diagrams

        generate_diagrams.main()
    except ImportError as e:
        print(f"Warning: Could not import diagram generator: {e}")
    except Exception as e:
        print(f"Error generating diagrams: {e}")
        raise


def run_surrogate_generator():
    """Run the surrogate coverage generator."""
    print("\n" + "=" * 60)
    print("Generating Surrogate Coverage")
    print("=" * 60)

    try:
        from docs._generators import generate_surrogates

        generate_surrogates.main()
    except ImportError as e:
        print(f"Warning: Could not import surrogate generator: {e}")
        print("  (This may be expected if sklearn is not installed)")
    except Exception as e:
        print(f"Error generating surrogate docs: {e}")
        # Don't raise - surrogates are optional


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate documentation assets for Surfaces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--catalogs",
        action="store_true",
        help="Generate function catalog tables",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization assets (requires plotly)",
    )
    parser.add_argument(
        "--diagrams",
        action="store_true",
        help="Generate architecture diagrams",
    )
    parser.add_argument(
        "--surrogates",
        action="store_true",
        help="Generate surrogate model coverage docs",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all assets (default if no specific flags)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove generated files before generating",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be generated without running",
    )

    args = parser.parse_args()

    # Default to all if nothing specified
    if not any([args.catalogs, args.plots, args.diagrams, args.surrogates, args.all]):
        args.all = True

    if args.dry_run:
        print("Dry run - would generate:")
        if args.all or args.catalogs:
            print("  - Function catalogs (RST tables)")
        if args.all or args.plots:
            print("  - Visualization assets (PNG plots)")
        if args.all or args.diagrams:
            print("  - Architecture diagrams (Mermaid/RST)")
        if args.all or args.surrogates:
            print("  - Surrogate model coverage (RST tables)")
        return 0

    if args.clean:
        clean_generated()

    # Ensure directories exist
    ensure_directories()

    # Run selected generators
    if args.all or args.catalogs:
        run_catalog_generator()

    if args.all or args.plots:
        run_plot_generator()

    if args.all or args.diagrams:
        run_diagram_generator()

    if args.all or args.surrogates:
        run_surrogate_generator()

    print("\n" + "=" * 60)
    print("Generation Complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
