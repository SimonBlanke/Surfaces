"""Test runner for all example files.

This module discovers and runs all Python example files in the examples/ directory.
Each example is run as a subprocess with a timeout to ensure isolation and prevent
blocking on interactive elements like fig.show().

Usage:
    pytest examples/test_examples.py -v
    python examples/test_examples.py  # Direct execution

Environment:
    SURFACES_TESTING=1 is set automatically to skip interactive elements.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def get_surfaces_root() -> Path:
    """Get the root directory of the Surfaces package."""
    return Path(__file__).parent.parent.resolve()


def find_all_python_examples() -> list[Path]:
    """Find all Python example files in the examples directory.

    Returns
    -------
    list[Path]
        Sorted list of paths to Python example files.
    """
    root = get_surfaces_root()
    examples_dir = root / "examples"

    if not examples_dir.exists():
        return []

    python_files = []
    for py_file in examples_dir.rglob("*.py"):
        # Skip test files, cache directories, and __init__ files
        if (
            "__pycache__" not in str(py_file)
            and ".pytest_cache" not in str(py_file)
            and not py_file.name.startswith("test_")
            and py_file.name != "__init__.py"
        ):
            python_files.append(py_file)

    return sorted(python_files)


def _example_id(example_path: Path) -> str:
    """Generate a test ID from an example path."""
    root = get_surfaces_root()
    try:
        return str(example_path.relative_to(root))
    except ValueError:
        return str(example_path)


# Collect examples at module load time for parametrization
ALL_EXAMPLES = find_all_python_examples()


@pytest.mark.parametrize("example_file", ALL_EXAMPLES, ids=_example_id)
def test_example_runs(example_file: Path):
    """Test that each example file executes without errors.

    Parameters
    ----------
    example_file : Path
        Path to the example file to test.
    """
    env = os.environ.copy()
    # Set testing flag to skip interactive elements like fig.show()
    env["SURFACES_TESTING"] = "1"
    # Ensure the package is importable
    env["PYTHONPATH"] = str(get_surfaces_root() / "src")

    result = subprocess.run(
        [sys.executable, str(example_file)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(example_file.parent),
        env=env,
    )

    if result.returncode != 0:
        error_msg = f"Example {example_file.name} failed to execute.\n"
        error_msg += f"stdout:\n{result.stdout}\n"
        error_msg += f"stderr:\n{result.stderr}"
        pytest.fail(error_msg)


def main():
    """Run all examples directly (not via pytest)."""
    examples = find_all_python_examples()
    print(f"Found {len(examples)} example files to test:\n")

    passed = 0
    failed = 0

    for example in examples:
        rel_path = _example_id(example)
        print(f"Running {rel_path}...", end=" ", flush=True)

        env = os.environ.copy()
        env["SURFACES_TESTING"] = "1"
        env["PYTHONPATH"] = str(get_surfaces_root() / "src")

        try:
            result = subprocess.run(
                [sys.executable, str(example)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(example.parent),
                env=env,
            )

            if result.returncode == 0:
                print("OK")
                passed += 1
            else:
                print("FAILED")
                print(f"  stdout: {result.stdout[:200]}...")
                print(f"  stderr: {result.stderr[:200]}...")
                failed += 1

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
