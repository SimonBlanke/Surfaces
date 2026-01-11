"""Test that README examples actually work.

Extracts Python code blocks from README.md and runs them.
This ensures documentation stays in sync with the actual API.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def extract_python_blocks(readme_path: Path) -> list[tuple[str, str]]:
    """Extract Python code blocks from README.md."""
    content = readme_path.read_text()

    # Pattern: <summary><b>Name</b></summary> or ### Name followed by ```python
    pattern = r'(?:<summary><b>([^<]+)</b></summary>|### ([^\n]+))\s*\n+```python\n(.*?)```'

    examples = []
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1) or match.group(2)
        code = match.group(3)
        name = name.strip().replace(" ", "_").lower()
        examples.append((name, code))

    return examples


# Get README path relative to this file
README_PATH = Path(__file__).parent.parent.parent / "README.md"
EXAMPLES = extract_python_blocks(README_PATH) if README_PATH.exists() else []

# Examples to skip (e.g., require external dependencies)
SKIP_EXAMPLES = {"integration_with_optimizers"}


@pytest.mark.parametrize("name,code", EXAMPLES, ids=[e[0] for e in EXAMPLES])
def test_readme_example(name: str, code: str):
    """Test that a README example runs without errors."""
    if name in SKIP_EXAMPLES:
        pytest.skip(f"Skipping {name} (external dependencies)")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        temp_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )

        assert result.returncode == 0, (
            f"Example '{name}' failed:\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    finally:
        temp_path.unlink()
