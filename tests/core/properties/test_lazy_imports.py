"""Static analysis: no optional dependencies imported at module level.

All modules under test_functions/ must be importable with only numpy installed.
Optional dependencies (sklearn, scipy, matplotlib, etc.) must use lazy imports
inside function/method bodies, never at module or class-body level.

This test uses AST parsing - no imports are actually executed.
"""

import ast
import sys
from pathlib import Path

import pytest

# Root of test_functions source
TEST_FUNCTIONS_ROOT = Path(__file__).resolve().parents[3] / "src" / "surfaces" / "test_functions"

# Only these non-stdlib packages are allowed at module level
ALLOWED_PACKAGES = {"numpy", "surfaces"}


def _is_allowed(module_name: str) -> bool:
    """Check if a top-level module name is allowed at module level."""
    if module_name in sys.stdlib_module_names:
        return True
    if module_name in ALLOWED_PACKAGES:
        return True
    return False


def _is_type_checking_guard(node: ast.If) -> bool:
    """Check if an If node is `if TYPE_CHECKING:`."""
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _collect_violations(node, filepath, violations):
    """Recursively collect imports that execute at import time.

    Skips function/method bodies (lazy imports) and TYPE_CHECKING guards.
    Recurses into class bodies, if/try/with/for blocks (all execute at import time).
    """
    for child in ast.iter_child_nodes(node):
        # Function bodies are lazy - skip entirely
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # TYPE_CHECKING guards are never executed at runtime
        if isinstance(child, ast.If) and _is_type_checking_guard(child):
            continue

        # Check imports
        if isinstance(child, ast.Import):
            for alias in child.names:
                module = alias.name.split(".")[0]
                if not _is_allowed(module):
                    violations.append((filepath, child.lineno, alias.name))

        elif isinstance(child, ast.ImportFrom):
            # Relative imports (level > 0) are always internal
            if child.level > 0:
                continue
            if child.module:
                module = child.module.split(".")[0]
                if not _is_allowed(module):
                    violations.append((filepath, child.lineno, child.module))

        # Recurse into compound statements (class, if, try, with, for, etc.)
        _collect_violations(child, filepath, violations)


def _scan_file(filepath):
    """Parse a single Python file and return import violations."""
    source = filepath.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(filepath))
    violations = []
    _collect_violations(tree, filepath, violations)
    return violations


def _collect_all_violations():
    """Scan all Python files under test_functions/ for violations."""
    all_violations = []
    for filepath in sorted(TEST_FUNCTIONS_ROOT.rglob("*.py")):
        all_violations.extend(_scan_file(filepath))
    return all_violations


class TestLazyImports:
    """Ensure optional dependencies use lazy imports in test_functions/."""

    def test_no_optional_imports_at_module_level(self):
        """All optional package imports must be inside function/method bodies.

        Modules under test_functions/ must be importable with only numpy
        installed. Any third-party import beyond numpy must be deferred
        to the function or method that actually uses it.
        """
        violations = _collect_all_violations()

        if violations:
            lines = [
                "Optional dependencies imported at module level in test_functions/.",
                "Move these imports into the functions/methods that use them.",
                "",
            ]
            for filepath, lineno, module in violations:
                rel = filepath.relative_to(TEST_FUNCTIONS_ROOT.parents[2])
                lines.append(f"  {rel}:{lineno} -> {module}")

            pytest.fail("\n".join(lines))
