#!/usr/bin/env python3
"""
Validate that tests/core/ does not import optional dependencies.

Core tests should only use minimal dependencies (numpy, pytest).
Tests requiring optional dependencies (sklearn, matplotlib, etc.) must go in tests/full/.

Exit codes:
    0: All tests valid
    1: Found forbidden imports in tests/core/
"""

import ast
import sys
from pathlib import Path
from typing import List, Set, Tuple

# Optional dependencies that are NOT allowed in tests/core/
FORBIDDEN_MODULES = {
    # ML dependencies
    "sklearn",
    "scikit-learn",
    "xgboost",
    # Visualization
    "matplotlib",
    "plotly",
    # Deep learning
    "tensorflow",
    "tf",
    "keras",
    "torch",
    "pytorch",
    # Time series
    "sktime",
    # Surrogates
    "onnxruntime",
    "onnx",
    # Other optional deps
    "pandas",
    "scipy",
}

# Allowed core dependencies
ALLOWED_MODULES = {
    "numpy",
    "np",
    "pytest",
    "surfaces",
    # Python standard library (partial list)
    "os",
    "sys",
    "pathlib",
    "tempfile",
    "threading",
    "time",
    "typing",
    "collections",
    "dataclasses",
    "functools",
    "itertools",
    "json",
    "re",
    "unittest",
    "warnings",
    "__future__",
}


def extract_imports_from_file(filepath: Path) -> Set[str]:
    """Extract all imported module names from a Python file.

    Parameters
    ----------
    filepath : Path
        Path to Python file to analyze.

    Returns
    -------
    Set[str]
        Set of module names imported in the file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=str(filepath))
    except SyntaxError as e:
        print(f"⚠️  Syntax error in {filepath}: {e}", file=sys.stderr)
        return set()

    imports = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Extract top-level module (e.g., "sklearn" from "sklearn.ensemble")
                top_level = alias.name.split(".")[0]
                imports.add(top_level)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # Extract top-level module
                top_level = node.module.split(".")[0]
                imports.add(top_level)

    return imports


def check_file(filepath: Path) -> List[str]:
    """Check a single file for forbidden imports.

    Parameters
    ----------
    filepath : Path
        Path to Python file to check.

    Returns
    -------
    List[str]
        List of forbidden module names found in the file.
    """
    imports = extract_imports_from_file(filepath)
    forbidden = []

    for module in imports:
        if module in FORBIDDEN_MODULES:
            forbidden.append(module)

    return forbidden


def validate_core_tests(core_tests_dir: Path) -> Tuple[bool, List[Tuple[Path, List[str]]]]:
    """Validate all Python files in tests/core/ directory.

    Parameters
    ----------
    core_tests_dir : Path
        Path to tests/core/ directory.

    Returns
    -------
    Tuple[bool, List[Tuple[Path, List[str]]]]
        (is_valid, violations) where violations is a list of (filepath, forbidden_modules).
    """
    if not core_tests_dir.exists():
        print(f"✅ Core tests directory does not exist: {core_tests_dir}")
        return True, []

    violations = []

    # Find all Python files in tests/core/
    for py_file in core_tests_dir.rglob("*.py"):
        # Skip __pycache__ and other generated files
        if "__pycache__" in py_file.parts or py_file.name.startswith("."):
            continue

        forbidden = check_file(py_file)
        if forbidden:
            violations.append((py_file, forbidden))

    return len(violations) == 0, violations


def main() -> int:
    """Main entry point."""
    # Find project root (assumes script is in scripts/ subdirectory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    core_tests_dir = project_root / "tests" / "core"

    print("🔍 Validating test dependencies...")
    print(f"   Core tests directory: {core_tests_dir}")
    print()

    is_valid, violations = validate_core_tests(core_tests_dir)

    if is_valid:
        print("✅ All core tests use only allowed dependencies!")
        return 0
    else:
        print("❌ Found forbidden imports in tests/core/:")
        print()
        for filepath, forbidden_modules in violations:
            rel_path = filepath.relative_to(project_root)
            print(f"   {rel_path}:")
            for module in forbidden_modules:
                print(f"      - {module}")
        print()
        print("📝 Forbidden modules in tests/core/:")
        print(f"   {', '.join(sorted(FORBIDDEN_MODULES))}")
        print()
        print("💡 Solution:")
        print("   Move tests requiring optional dependencies to tests/full/")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
