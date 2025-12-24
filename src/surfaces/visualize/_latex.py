# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""LaTeX/PDF plot generation for 2D algebraic test functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..test_functions._base_test_function import BaseTestFunction

from ._errors import MissingDependencyError, PlotCompatibilityError
from ._utils import validate_plot

_LATEX_TEMPLATE = r"""\documentclass[border=10pt]{{standalone}}
\usepackage{{pgfplots}}
\usepackage{{amsmath}}
\usepackage{{amssymb}}

\pgfplotsset{{compat=1.18}}

% Define the test function
\pgfmathdeclarefunction{{{func_id}}}{{2}}{{%
  \pgfmathparse{{{pgfmath_formula}}}%
}}

\begin{{document}}

\begin{{tikzpicture}}
  \begin{{axis}}[
    width=14cm,
    height=12cm,
    view={{{view_azimuth}}}{{{view_elevation}}},
    xlabel={{$x$}},
    ylabel={{$y$}},
    zlabel={{$f(x,y)$}},
    xlabel style={{font=\large, sloped}},
    ylabel style={{font=\large, sloped}},
    zlabel style={{font=\large, rotate=-90}},
    title={{\textbf{{\Large {title}}}}},
    title style={{yshift=5pt}},
    colormap={{jet}}{{
      rgb255=(128,0,0)
      rgb255=(255,0,0)
      rgb255=(255,128,0)
      rgb255=(255,255,0)
      rgb255=(128,255,128)
      rgb255=(0,255,255)
      rgb255=(0,128,255)
      rgb255=(0,0,255)
      rgb255=(0,0,128)
    }},
    colorbar,
    colorbar style={{
      ylabel={{$f(x,y)$}},
      ylabel style={{font=\normalsize, rotate=-90}},
    }},
    domain={domain_min}:{domain_max},
    y domain={domain_min}:{domain_max},
    samples={samples},
    samples y={samples},
    z buffer=sort,
    mesh/ordering=y varies,
    shader=interp,
    {zmin_line}
    {zmax_line}
  ]
    \addplot3[surf, opacity=0.95] {{{func_id}(x,y)}};
  \end{{axis}}

  % Formula box below the plot
  \node[anchor=north, yshift=-0.5cm, align=center] at (current bounding box.south) {{
    \fbox{{
      \parbox{{12cm}}{{
        \centering
        \vspace{{0.3cm}}
        $\displaystyle {latex_formula}$
        \vspace{{0.2cm}}

        \small {global_minimum_text}
        \vspace{{0.2cm}}
      }}
    }}
  }};
\end{{tikzpicture}}

\end{{document}}
"""


def plot_latex(
    func: "BaseTestFunction",
    output_path: Optional[str] = None,
    compile_pdf: bool = False,
    samples: int = 100,
    view_azimuth: int = -35,
    view_elevation: int = 25,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    title: Optional[str] = None,
) -> str:
    """Generate a publication-quality LaTeX file with pgfplots 3D surface.

    Creates a standalone LaTeX document with a 3D surface plot and the
    mathematical formula in a box below. Requires the function to have
    `latex_formula` and `pgfmath_formula` attributes.

    Args:
        func: A 2-dimensional algebraic test function with formula attributes.
        output_path: Path for the output .tex file. Defaults to
            '{function_name}.tex' in the current directory.
        compile_pdf: If True, compile the .tex to PDF using pdflatex.
            Requires pdflatex to be installed.
        samples: Number of samples per axis for the surface (default: 100).
            Higher values give smoother surfaces but slower compilation.
        view_azimuth: Horizontal viewing angle in degrees (default: -35).
        view_elevation: Vertical viewing angle in degrees (default: 25).
        zmin: Optional minimum z-axis value. Auto-scaled if None.
        zmax: Optional maximum z-axis value. Auto-scaled if None.
        title: Plot title. Defaults to function name.

    Returns:
        Path to the generated .tex file (or .pdf if compile_pdf=True).

    Raises:
        PlotCompatibilityError: If function is not 2D or lacks formula attributes.
        MissingDependencyError: If compile_pdf=True but pdflatex not found.
        RuntimeError: If PDF compilation fails.

    Examples:
        >>> from surfaces.test_functions import AckleyFunction
        >>> from surfaces.visualize import plot_latex
        >>> func = AckleyFunction()
        >>> tex_path = plot_latex(func)
        >>> print(f"Generated: {tex_path}")
        Generated: ackley_function.tex

        >>> # Compile to PDF
        >>> pdf_path = plot_latex(func, compile_pdf=True)
        >>> print(f"Generated: {pdf_path}")
        Generated: ackley_function.pdf
    """
    validate_plot(func, "latex")

    # Check for required attributes
    latex_formula = getattr(func, "latex_formula", None)
    pgfmath_formula = getattr(func, "pgfmath_formula", None)

    if latex_formula is None:
        raise PlotCompatibilityError(
            plot_name="latex",
            reason="function has no 'latex_formula' attribute",
            func=func,
            suggestions=["Use algebraic test functions which have formula attributes"],
        )

    if pgfmath_formula is None:
        raise PlotCompatibilityError(
            plot_name="latex",
            reason="function has no 'pgfmath_formula' attribute (some complex functions cannot be rendered in pgfplots)",
            func=func,
            suggestions=["Use plot_surface() for interactive visualization instead"],
        )

    # Get function metadata
    func_name = getattr(func, "name", type(func).__name__)
    func_id = getattr(func, "_name_", type(func).__name__.lower())
    default_bounds = getattr(func, "default_bounds", (-5.0, 5.0))
    f_global = getattr(func, "f_global", None)
    x_global = getattr(func, "x_global", None)

    # Build global minimum text
    if f_global is not None and x_global is not None:
        # Handle multiple global minima
        if hasattr(x_global, "ndim") and x_global.ndim == 2:
            x_str = r"\pm " + ", ".join(
                f"{abs(x_global[0, i]):.4g}" for i in range(x_global.shape[1])
            )
            global_minimum_text = f"Global minimum: $f({x_str}) = {f_global}$"
        else:
            x_str = ", ".join(f"{x:.4g}" for x in x_global)
            global_minimum_text = f"Global minimum: $f({x_str}) = {f_global}$"
    else:
        global_minimum_text = ""

    # Build z-axis limits
    zmin_line = f"zmin={zmin}," if zmin is not None else ""
    zmax_line = f"zmax={zmax}," if zmax is not None else ""

    # Generate LaTeX content
    latex_content = _LATEX_TEMPLATE.format(
        func_id=func_id,
        pgfmath_formula=pgfmath_formula,
        title=title or func_name,
        domain_min=default_bounds[0],
        domain_max=default_bounds[1],
        samples=samples,
        view_azimuth=view_azimuth,
        view_elevation=view_elevation,
        zmin_line=zmin_line,
        zmax_line=zmax_line,
        latex_formula=latex_formula,
        global_minimum_text=global_minimum_text,
    )

    # Determine output path
    if output_path is None:
        output_path = f"{func_id}.tex"

    # Ensure .tex extension
    if not output_path.endswith(".tex"):
        output_path = output_path + ".tex"

    # Write the file
    with open(output_path, "w") as f:
        f.write(latex_content)

    # Optionally compile to PDF
    if compile_pdf:
        import shutil
        import subprocess

        if shutil.which("pdflatex") is None:
            raise MissingDependencyError(
                ["pdflatex"],
                "PDF compilation requires pdflatex. Install TeX Live or MiKTeX.",
            )

        # Get directory and filename
        tex_dir = os.path.dirname(output_path) or "."
        tex_file = os.path.basename(output_path)

        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file],
                cwd=tex_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"pdflatex compilation failed:\n{result.stdout}\n{result.stderr}"
                )

            # Return PDF path
            pdf_path = output_path.replace(".tex", ".pdf")
            return pdf_path

        except subprocess.TimeoutExpired:
            raise RuntimeError("pdflatex compilation timed out after 120 seconds")

    return output_path
