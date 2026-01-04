# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Test coverage report for Surfaces test functions.

This module provides utilities to generate a report of which test functions
have dedicated tests and which are only covered by generic parametrized tests.

Run with: python -m tests.test_coverage_report
"""

from pathlib import Path
from typing import Dict, List, Set


def get_all_function_classes() -> Dict[str, List[type]]:
    """Get all test function classes organized by category."""
    result = {}

    # Algebraic functions
    try:
        from surfaces.test_functions.algebraic import (
            algebraic_functions,
            algebraic_functions_1d,
            algebraic_functions_2d,
            algebraic_functions_nd,
        )

        result["Algebraic (All)"] = algebraic_functions
        result["Algebraic (1D)"] = algebraic_functions_1d
        result["Algebraic (2D)"] = algebraic_functions_2d
        result["Algebraic (ND)"] = algebraic_functions_nd
    except ImportError:
        pass

    # Engineering functions
    try:
        from surfaces.test_functions.engineering import engineering_functions

        result["Engineering"] = engineering_functions
    except ImportError:
        pass

    # BBOB functions
    try:
        from surfaces.test_functions.bbob import BBOB_FUNCTIONS

        result["BBOB"] = list(BBOB_FUNCTIONS.values())
    except ImportError:
        pass

    # CEC 2013 functions
    try:
        from tests.conftest import CEC2013_FUNCTIONS

        if CEC2013_FUNCTIONS:
            result["CEC 2013"] = CEC2013_FUNCTIONS
    except ImportError:
        pass

    # CEC 2014 functions
    try:
        from tests.conftest import CEC2014_FUNCTIONS

        if CEC2014_FUNCTIONS:
            result["CEC 2014"] = CEC2014_FUNCTIONS
    except ImportError:
        pass

    # CEC 2017 functions
    try:
        from tests.conftest import CEC2017_FUNCTIONS

        if CEC2017_FUNCTIONS:
            result["CEC 2017"] = CEC2017_FUNCTIONS
    except ImportError:
        pass

    # ML functions
    try:
        from surfaces.test_functions.machine_learning import machine_learning_functions

        if machine_learning_functions:
            result["Machine Learning"] = machine_learning_functions
    except ImportError:
        pass

    return result


def get_test_files() -> List[Path]:
    """Get all test files in the tests directory."""
    tests_dir = Path(__file__).parent
    return list(tests_dir.rglob("test_*.py"))


def find_function_mentions(test_files: List[Path]) -> Set[str]:
    """Find all function class names mentioned in test files."""
    mentions = set()
    for test_file in test_files:
        if test_file.name == "test_coverage_report.py":
            continue  # Skip this file
        content = test_file.read_text()
        # Look for class names in imports and test code
        for line in content.split("\n"):
            # Simple heuristic: look for CamelCase words that end with "Function"
            words = line.replace(",", " ").replace("(", " ").replace(")", " ").split()
            for word in words:
                if word.endswith("Function") and word[0].isupper():
                    mentions.add(word)
    return mentions


def generate_report() -> str:
    """Generate the coverage report."""
    categories = get_all_function_classes()
    test_files = get_test_files()
    mentioned = find_function_mentions(test_files)

    lines = []
    lines.append("=" * 70)
    lines.append("SURFACES TEST FUNCTION COVERAGE REPORT")
    lines.append("=" * 70)
    lines.append("")

    total_functions = 0
    total_mentioned = 0

    for category, functions in categories.items():
        lines.append(f"\n{category}")
        lines.append("-" * len(category))

        category_total = len(functions)
        category_mentioned = 0

        for func in functions:
            name = func.__name__
            is_mentioned = name in mentioned
            status = "[x]" if is_mentioned else "[ ]"
            lines.append(f"  {status} {name}")

            if is_mentioned:
                category_mentioned += 1

        lines.append(
            f"\n  Coverage: {category_mentioned}/{category_total} "
            f"({100*category_mentioned/category_total:.1f}%)"
        )

        total_functions += category_total
        total_mentioned += category_mentioned

    lines.append("")
    lines.append("=" * 70)
    lines.append(
        f"TOTAL: {total_mentioned}/{total_functions} functions mentioned in tests "
        f"({100*total_mentioned/total_functions:.1f}%)"
    )
    lines.append("=" * 70)

    lines.append("")
    lines.append("Note: [x] means the function is explicitly mentioned in test files.")
    lines.append("      [ ] means it's only covered by parametrized tests.")
    lines.append("      Both types are tested - this report shows explicit coverage.")

    return "\n".join(lines)


def generate_test_structure_report() -> str:
    """Generate a report of the test file structure."""
    tests_dir = Path(__file__).parent
    lines = []

    lines.append("")
    lines.append("=" * 70)
    lines.append("TEST FILE STRUCTURE")
    lines.append("=" * 70)
    lines.append("")

    def show_tree(path: Path, prefix: str = "") -> None:
        """Recursively show directory tree."""
        if path.is_file():
            if path.suffix == ".py" and not path.name.startswith("_"):
                lines.append(f"{prefix}{path.name}")
        else:
            lines.append(f"{prefix}{path.name}/")
            children = sorted(path.iterdir())
            for i, child in enumerate(children):
                is_last = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "    ")
                show_tree(child, child_prefix)

    show_tree(tests_dir)

    return "\n".join(lines)


def main():
    """Run the coverage report."""
    print(generate_report())
    print(generate_test_structure_report())


if __name__ == "__main__":
    main()


# =============================================================================
# Pytest Tests for Coverage Validation
# =============================================================================


class TestCoverageValidation:
    """Validate that key function categories have test coverage."""

    def test_algebraic_functions_covered(self):
        """All algebraic functions should be in test parametrization."""
        from surfaces.test_functions.algebraic import algebraic_functions
        from tests.conftest import algebraic_functions as conftest_algebraic

        # conftest should have all algebraic functions
        assert set(algebraic_functions) == set(conftest_algebraic)

    def test_engineering_functions_covered(self):
        """All engineering functions should be in test parametrization."""
        from surfaces.test_functions.engineering import engineering_functions
        from tests.conftest import engineering_functions as conftest_engineering

        assert set(engineering_functions) == set(conftest_engineering)

    def test_bbob_functions_covered(self):
        """All BBOB functions should be in test parametrization."""
        from surfaces.test_functions.bbob import BBOB_FUNCTIONS
        from tests.conftest import BBOB_FUNCTION_LIST

        assert set(BBOB_FUNCTIONS.values()) == set(BBOB_FUNCTION_LIST)

    def test_conftest_has_all_categories(self):
        """conftest should export all major function categories."""
        from tests import conftest

        # Check required exports exist
        assert hasattr(conftest, "algebraic_functions")
        assert hasattr(conftest, "engineering_functions")
        assert hasattr(conftest, "BBOB_FUNCTION_LIST")
        assert hasattr(conftest, "CEC2014_FUNCTIONS")
        assert hasattr(conftest, "machine_learning_functions")

    def test_instantiate_helper_works(self):
        """instantiate_function helper should work for all types."""
        from surfaces.test_functions import BealeFunction, SphereFunction
        from tests.conftest import instantiate_function

        # ND function with n_dim
        func1 = instantiate_function(SphereFunction, n_dim=5)
        assert len(func1.search_space) == 5

        # 2D function without n_dim
        func2 = instantiate_function(BealeFunction)
        assert len(func2.search_space) == 2
