"""Import isolation tests for ML function files.

Uses AST parsing to statically analyze source files and ensure that
optional dependencies (xgboost, tensorflow, sktime, ...) are never
imported at module level in ML function files.

This prevents the situation where installing surfaces[ml] breaks
because a single file pulls in an uninstalled optional package,
taking down the entire classification/regression import chain.

What is checked
---------------
| Test Class                     | What it catches                        |
|--------------------------------|----------------------------------------|
| TestNoTopLevelOptionalImports  | `from xgboost import ...` at file top  |
| TestOptionalDepGuards          | Missing _check_*() guard function for  |
|                                | files that use optional dependencies    |

Correct patterns for optional dependencies
------------------------------------------
Pattern A -- _dependencies class attribute (preferred):

    class MyXGBoostFunction(BaseClassification):
        _dependencies = ("scikit-learn", "xgboost")

        def _ml_objective(self):
            from xgboost import XGBClassifier        # Lazy import
            ...

    BaseTestFunction.__init__ calls _check_dependencies() which reads
    _dependencies and calls check_dependency() for each entry.
    MachineLearningFunction overrides _check_dependencies() to skip
    the check when use_surrogate=True.

Pattern B -- centralised utility (direct call):

    from surfaces._dependencies import check_dependency

    class MyXGBoostFunction(BaseClassification):
        def __init__(self, ...):
            check_dependency("xgboost", "xgboost")  # Fail early
            ...

        def _ml_objective(self):
            from xgboost import XGBClassifier        # Lazy import
            ...

Pattern C -- local guard function:

    def _check_xgboost():
        try:
            import xgboost
            return True
        except ImportError:
            raise ImportError(
                "XGBoost classifier requires xgboost. "
                "Install with: pip install surfaces[xgboost]"
            )

    class MyXGBoostFunction(BaseClassification):
        def __init__(self, ...):
            _check_xgboost()           # Fail early with clear message
            ...

        def _ml_objective(self):
            from xgboost import XGBClassifier   # Lazy import here
            ...
"""

import ast
from pathlib import Path

import pytest

import surfaces.test_functions.machine_learning as _ml_root

_ML_BASE = Path(_ml_root.__file__).parent

# Packages that are NOT part of the base `surfaces[ml]` install and must
# not be imported at module level. sklearn IS the base ML dep and is fine.
_OPTIONAL_PACKAGES = frozenset(
    {
        "xgboost",
        "tensorflow",
        "keras",
        "sktime",
        "torch",
        "gymnasium",
        "gym",
    }
)


def _get_ml_function_files():
    """Find all concrete ML function files (not __init__, not _base_*)."""
    files = []
    for p in _ML_BASE.rglob("test_functions/*.py"):
        if p.name == "__init__.py" or p.name.startswith("_"):
            continue
        files.append(p)
    return sorted(files)


def _analyze_imports(filepath):
    """Parse a Python file and categorize its imports.

    "Module-level" means any import that is NOT inside a function body.
    This includes imports inside try/except, if/else, and class bodies
    at module scope, since all of these execute at import time.

    Only imports inside FunctionDef or AsyncFunctionDef are considered
    function-level (lazy imports that execute on call, not on import).

    Example AST structure::

        Module                       <- module scope
          Try                        <- module scope (executes at import)
            ImportFrom(xgboost)      <- module-level -> CAUGHT
          FunctionDef(_check_xgb)    <- function scope boundary
            Try
              Import(xgboost)        <- function-level -> OK (lazy)
          ClassDef(MyFunction)       <- module scope
            FunctionDef(method)      <- function scope boundary
              ImportFrom(xgboost)    <- function-level -> OK (lazy)

    Returns
    -------
    module_level : set
        Package names imported at module level (outside any function).
    all_imports : set
        Package names imported anywhere in the file.
    """
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    module_level = set()
    all_imports = set()

    def _extract_package(node):
        """Extract the root package name from an import node."""
        if isinstance(node, ast.Import):
            return {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            return {node.module.split(".")[0]}
        return set()

    def _walk(node, inside_function=False):
        """Recursively walk AST, tracking whether we're inside a function."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            inside_function = True

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            pkgs = _extract_package(node)
            all_imports.update(pkgs)
            if not inside_function:
                module_level.update(pkgs)

        for child in ast.iter_child_nodes(node):
            _walk(child, inside_function)

    _walk(tree)
    return module_level, all_imports


def _has_check_guard(filepath):
    """Check if the file has an early dependency guard.

    Recognises three patterns:
    1. A local ``def _check_*():`` guard function.
    2. A call to the centralised ``check_dependency(...)`` utility.
    3. A class-level ``_dependencies = {...}`` attribute (centralised base-class check).
    """
    source = filepath.read_text()
    tree = ast.parse(source, filename=str(filepath))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_check_"):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "check_dependency":
                return True
            if isinstance(func, ast.Attribute) and func.attr == "check_dependency":
                return True
        # Pattern 3: _dependencies = (...) or _dependencies = {...} class attribute
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_dependencies":
                    if isinstance(node.value, (ast.Dict, ast.Tuple)):
                        return True
    return False


def _file_id(filepath):
    """Generate a readable test ID from a file path."""
    return filepath.relative_to(_ML_BASE).as_posix()


_ML_FUNCTION_FILES = _get_ml_function_files()


@pytest.mark.ml
@pytest.mark.static
class TestNoTopLevelOptionalImports:
    """Optional packages must not be imported at module level.

    Module-level imports execute when the package is first imported.
    If any file in tabular/classification/test_functions/ does
    `from xgboost import XGBClassifier` at the top level, then importing
    ANY classifier (even SVM or RandomForest) will fail when xgboost
    is not installed.

    The correct pattern is:
    - Lazy import inside _ml_objective()
    - _check_*() guard in __init__() for early error reporting
    """

    @pytest.mark.parametrize("filepath", _ML_FUNCTION_FILES, ids=_file_id)
    def test_no_optional_top_level_import(self, filepath):
        """No ML function file may import optional packages at module level."""
        top_level, _ = _analyze_imports(filepath)
        bad = top_level & _OPTIONAL_PACKAGES

        assert not bad, (
            f"{filepath.name}: Top-level import of optional packages: {sorted(bad)}\n"
            f"Move these imports inside _ml_objective() and add\n"
            f"a _check_*() guard in __init__(). See xgboost_image_classifier.py\n"
            f"for the correct pattern."
        )


# Files with optional-dep usage but no _check_*() guard (pre-existing,
# tracked for cleanup). New files MUST NOT be added to this set.
_KNOWN_MISSING_GUARDS = {
    "image_augmentation.py",
    "cnn_keras_nas.py",
    "mlp_pytorch_nas.py",
    "dqn_cartpole.py",
    "mobilenet_transfer_learning.py",
}


@pytest.mark.ml
@pytest.mark.static
class TestOptionalDepGuards:
    """Files using optional deps must have a _check_*() guard function.

    The guard function is called in __init__() and raises a clear
    ImportError if the optional package is not installed. Without it,
    users get a confusing error from the lazy import deep inside
    _ml_objective() instead of a helpful message at
    instantiation time.
    """

    @pytest.mark.parametrize("filepath", _ML_FUNCTION_FILES, ids=_file_id)
    def test_has_guard_if_uses_optional_dep(self, filepath):
        """Files importing optional packages must define _check_*()."""
        _, all_imports = _analyze_imports(filepath)
        optional_used = all_imports & _OPTIONAL_PACKAGES

        if not optional_used:
            pytest.skip("File does not use optional dependencies")

        if filepath.name in _KNOWN_MISSING_GUARDS:
            pytest.xfail(f"{filepath.name}: Known missing _check_*() guard (tracked for cleanup)")

        assert _has_check_guard(filepath), (
            f"{filepath.name}: Uses optional packages {sorted(optional_used)} "
            f"but does not define a _check_*() guard function.\n"
            f"Add a function like:\n"
            f"    def _check_{sorted(optional_used)[0]}():\n"
            f"        try:\n"
            f"            import {sorted(optional_used)[0]}\n"
            f"            return True\n"
            f"        except ImportError:\n"
            f'            raise ImportError("... Install with: pip install surfaces[...]")'
        )
