#!/usr/bin/env python3
"""Convert mermaid diagrams in RST/MD files to static SVG files."""

import re
import subprocess
from pathlib import Path

DIAGRAMS_DIR = Path(__file__).parent.parent / "source/_generated/diagrams"
SVG_OUTPUT_DIR = Path(__file__).parent.parent / "source/_static/diagrams"


def extract_mermaid_from_rst(content: str) -> str | None:
    """Extract mermaid code from RST file with .. mermaid:: directive."""
    pattern = r"\.\. mermaid::\s*\n((?:\n|    .+\n)+)"
    match = re.search(pattern, content)
    if match:
        # Remove the 4-space indentation
        lines = match.group(1).split("\n")
        dedented = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return dedented.strip()
    return None


def extract_mermaid_from_md(content: str) -> str | None:
    """Extract mermaid code from MD file with ```mermaid block."""
    pattern = r"```mermaid\n(.*?)```"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def generate_svg(mermaid_code: str, output_path: Path) -> bool:
    """Generate SVG from mermaid code using mmdc CLI."""
    # Write temp mermaid file
    temp_mmd = output_path.with_suffix(".mmd")
    temp_mmd.write_text(mermaid_code)

    try:
        result = subprocess.run(
            [
                "mmdc",
                "-i",
                str(temp_mmd),
                "-o",
                str(output_path),
                "-b",
                "transparent",
                "-p",
                "/tmp/puppeteer-config.json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"  Error: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("  Error: mmdc timed out")
        return False
    finally:
        temp_mmd.unlink(missing_ok=True)


def update_rst_file(file_path: Path, svg_filename: str) -> str:
    """Update RST file to use image reference instead of mermaid directive."""
    content = file_path.read_text()

    # Pattern to match the mermaid directive and its content
    pattern = r"\.\. mermaid::\s*\n((?:\n|    .+\n)+)"

    # Replacement with image directive
    replacement = f""".. image:: /_static/diagrams/{svg_filename}
   :alt: {file_path.stem.replace('_', ' ').title()} Diagram
   :align: center
   :class: diagram

"""
    return re.sub(pattern, replacement, content)


def update_md_file(file_path: Path, svg_filename: str) -> str:
    """Update MD file to use image reference instead of mermaid block."""
    content = file_path.read_text()

    # Pattern to match the mermaid code block
    pattern = r"```mermaid\n.*?```"

    # Replacement with image reference (MyST markdown syntax)
    replacement = f"""```{{image}} /_static/diagrams/{svg_filename}
:alt: {file_path.stem.replace('_', ' ').title()} Diagram
:align: center
:class: diagram
```"""
    return re.sub(pattern, replacement, content, flags=re.DOTALL)


def main():
    """Convert all mermaid diagrams to SVG."""
    SVG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all files with mermaid content
    rst_files = list(DIAGRAMS_DIR.glob("*.rst"))
    md_files = list(DIAGRAMS_DIR.glob("*.md"))

    converted = 0
    for file_path in rst_files + md_files:
        content = file_path.read_text()

        # Check if file has mermaid content
        if file_path.suffix == ".rst":
            mermaid_code = extract_mermaid_from_rst(content)
        else:
            mermaid_code = extract_mermaid_from_md(content)

        if not mermaid_code:
            continue

        svg_filename = f"{file_path.stem}.svg"
        svg_path = SVG_OUTPUT_DIR / svg_filename

        print(f"Converting {file_path.name} -> {svg_filename}")

        if generate_svg(mermaid_code, svg_path):
            # Update the source file
            if file_path.suffix == ".rst":
                new_content = update_rst_file(file_path, svg_filename)
            else:
                new_content = update_md_file(file_path, svg_filename)

            file_path.write_text(new_content)
            converted += 1
            print("  Done")
        else:
            print("  Failed to generate SVG")

    print(f"\nConverted {converted} diagrams")


if __name__ == "__main__":
    main()
