"""Test documentation code snippets.

This module extracts and tests Python code blocks from RST documentation files.
Each code block is run as a subprocess to ensure isolation and catch errors.

Usage:
    pytest docs/tests/test_doc_snippets.py -v
    python docs/tests/test_doc_snippets.py  # Direct execution
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def get_docs_root() -> Path:
    """Get the root directory of the documentation source."""
    return Path(__file__).parent.parent / "source"


def get_surfaces_root() -> Path:
    """Get the root directory of the Surfaces package."""
    return Path(__file__).parent.parent.parent.resolve()


# Directories to test (relative to docs/source)
TESTABLE_DIRS = [
    "examples/getting_started",
    "examples/test_functions",
    "examples/integrations",
    "examples/visualization_examples",
]


def extract_code_blocks(rst_file: Path) -> list[tuple[str, int]]:
    """Extract Python code blocks from an RST file.

    Parameters
    ----------
    rst_file : Path
        Path to the RST file.

    Returns
    -------
    list[tuple[str, int]]
        List of (code, line_number) tuples.
    """
    content = rst_file.read_text()
    blocks = []

    # Pattern for code-block:: python
    pattern = r"^\.\. code-block:: python\s*\n((?:\n|[ \t]+.+\n)+)"

    for match in re.finditer(pattern, content, re.MULTILINE):
        # Get line number
        line_num = content[: match.start()].count("\n") + 1

        # Extract and dedent the code block
        code_block = match.group(1)
        lines = code_block.split("\n")

        # Find minimum indentation (excluding empty lines)
        non_empty_lines = [line for line in lines if line.strip()]
        if not non_empty_lines:
            continue

        min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

        # Remove common indentation
        dedented_lines = []
        for line in lines:
            if line.strip():
                dedented_lines.append(line[min_indent:])
            else:
                dedented_lines.append("")

        code = "\n".join(dedented_lines).strip()
        if code:
            blocks.append((code, line_num))

    return blocks


def find_testable_rst_files() -> list[Path]:
    """Find all RST files in testable directories.

    Returns
    -------
    list[Path]
        Sorted list of RST file paths.
    """
    docs_root = get_docs_root()
    rst_files = []

    for test_dir in TESTABLE_DIRS:
        dir_path = docs_root / test_dir
        if dir_path.exists():
            for rst_file in dir_path.rglob("*.rst"):
                # Skip index files as they typically don't have runnable code
                if rst_file.name != "index.rst":
                    rst_files.append(rst_file)

    return sorted(rst_files)


def get_all_code_blocks() -> list[tuple[Path, str, int]]:
    """Get all code blocks from all testable RST files.

    Returns
    -------
    list[tuple[Path, str, int]]
        List of (rst_file, code, line_number) tuples.
    """
    all_blocks = []

    for rst_file in find_testable_rst_files():
        blocks = extract_code_blocks(rst_file)
        for code, line_num in blocks:
            all_blocks.append((rst_file, code, line_num))

    return all_blocks


def _block_id(block: tuple[Path, str, int]) -> str:
    """Generate a test ID for a code block."""
    rst_file, code, line_num = block
    docs_root = get_docs_root()
    try:
        rel_path = rst_file.relative_to(docs_root)
    except ValueError:
        rel_path = rst_file.name
    return f"{rel_path}:L{line_num}"


# Collect all code blocks at module load time
ALL_CODE_BLOCKS = get_all_code_blocks()


@pytest.mark.parametrize("code_block", ALL_CODE_BLOCKS, ids=_block_id)
def test_code_block_runs(code_block: tuple[Path, str, int]):
    """Test that each code block executes without errors.

    Parameters
    ----------
    code_block : tuple[Path, str, int]
        Tuple of (rst_file, code, line_number).
    """
    rst_file, code, line_num = code_block

    # Skip blocks that are clearly incomplete snippets
    if "..." in code or "# ..." in code:
        pytest.skip("Incomplete snippet")

    # Create a temporary file with the code
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as tmp_file:
        tmp_file.write(code)
        tmp_path = tmp_file.name

    try:
        env = os.environ.copy()
        env["SURFACES_TESTING"] = "1"
        env["PYTHONPATH"] = str(get_surfaces_root() / "src")

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(get_surfaces_root()),
            env=env,
        )

        if result.returncode != 0:
            error_msg = f"Code block at {rst_file.name}:L{line_num} failed.\n"
            error_msg += f"Code:\n{code[:500]}...\n\n"
            error_msg += f"stdout:\n{result.stdout}\n"
            error_msg += f"stderr:\n{result.stderr}"
            pytest.fail(error_msg)

    finally:
        os.unlink(tmp_path)


def main():
    """Run all code block tests directly (not via pytest)."""
    blocks = get_all_code_blocks()
    print(f"Found {len(blocks)} code blocks to test:\n")

    passed = 0
    failed = 0
    skipped = 0

    for rst_file, code, line_num in blocks:
        block_id = _block_id((rst_file, code, line_num))
        print(f"Testing {block_id}...", end=" ", flush=True)

        # Skip incomplete snippets
        if "..." in code or "# ..." in code:
            print("SKIPPED (incomplete)")
            skipped += 1
            continue

        # Create temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp_file:
            tmp_file.write(code)
            tmp_path = tmp_file.name

        try:
            env = os.environ.copy()
            env["SURFACES_TESTING"] = "1"
            env["PYTHONPATH"] = str(get_surfaces_root() / "src")

            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(get_surfaces_root()),
                env=env,
            )

            if result.returncode == 0:
                print("OK")
                passed += 1
            else:
                print("FAILED")
                print(f"  stderr: {result.stderr[:200]}...")
                failed += 1

        except subprocess.TimeoutExpired:
            print("TIMEOUT")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
        finally:
            os.unlink(tmp_path)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
