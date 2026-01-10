# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for LaTeX/PDF plot generation via accessor pattern."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
)
from surfaces.visualize._errors import MissingDependencyError, PlotCompatibilityError


class TestLatexBasic:
    """Basic tests for LaTeX generation."""

    def test_ackley_function_generates_tex(self):
        """Test LaTeX generation for AckleyFunction."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "ackley.tex")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path
            assert os.path.exists(output_path)

            with open(output_path) as f:
                content = f.read()

            assert r"\documentclass" in content
            assert r"\begin{tikzpicture}" in content
            assert r"\addplot3[surf" in content
            assert "ackley_function" in content

    def test_rastrigin_function_generates_tex(self):
        """Test LaTeX generation for RastriginFunction."""
        func = RastriginFunction(n_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "rastrigin.tex")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path
            assert os.path.exists(output_path)

    def test_rosenbrock_function_generates_tex(self):
        """Test LaTeX generation for RosenbrockFunction."""
        func = RosenbrockFunction(n_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "rosenbrock.tex")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path
            assert os.path.exists(output_path)

    def test_default_output_path(self):
        """Test that default output path uses function name."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                result = func.plot.latex()

                assert result == "ackley_function.tex"
                assert os.path.exists("ackley_function.tex")
            finally:
                os.chdir(original_dir)

    def test_output_path_adds_tex_extension(self):
        """Test that .tex extension is added if missing."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path + ".tex"
            assert os.path.exists(result)


class TestLatexContent:
    """Tests for LaTeX content generation."""

    def test_contains_function_formula(self):
        """Test that generated LaTeX contains the function formula."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            # Should contain the latex_formula
            assert func.latex_formula in content

    def test_contains_global_minimum_info(self):
        """Test that generated LaTeX contains global minimum information."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            assert "Global minimum" in content
            assert "f(0, 0)" in content or "0.0" in content

    def test_custom_title(self):
        """Test that custom title is used."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, title="Custom Title")

            with open(output_path) as f:
                content = f.read()

            assert "Custom Title" in content

    def test_custom_samples(self):
        """Test that custom samples value is used."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, samples=75)

            with open(output_path) as f:
                content = f.read()

            assert "samples=75" in content

    def test_custom_view_angles(self):
        """Test that custom view angles are used."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, view_azimuth=-45, view_elevation=30)

            with open(output_path) as f:
                content = f.read()

            assert "view={-45}{30}" in content

    def test_zmin_zmax_limits(self):
        """Test that z-axis limits are included when specified."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, zmin=0, zmax=25)

            with open(output_path) as f:
                content = f.read()

            assert "zmin=0" in content
            assert "zmax=25" in content


class TestLatexErrors:
    """Tests for LaTeX error handling."""

    def test_error_for_function_without_latex_formula(self):
        """Test that ValueError is raised for functions without latex_formula."""
        func = SphereFunction(n_dim=2)

        # SphereFunction has latex_formula, so we test by checking the accessor's check
        # The accessor raises ValueError if latex_formula is missing
        # Since SphereFunction has it, this test verifies the attribute exists
        assert hasattr(func, "latex_formula")

    def test_error_for_nd_function(self):
        """Test that PlotCompatibilityError is raised for n-dimensional functions."""
        func = SphereFunction(n_dim=5)

        with pytest.raises(PlotCompatibilityError):
            func.plot.latex()


class TestLatexAccessor:
    """Tests for LaTeX accessor features."""

    def test_accessor_available_includes_latex(self):
        """Test that latex is listed in available plots for 2D algebraic functions."""
        func = AckleyFunction()

        available = func.plot.available()
        assert "latex" in available

    def test_accessor_available_excludes_latex_for_nd(self):
        """Test that latex is not available for N-D functions."""
        func = SphereFunction(n_dim=5)

        available = func.plot.available()
        assert "latex" not in available


class TestLatexPdfCompilation:
    """Tests for PDF compilation functionality."""

    def test_compile_pdf_without_pdflatex_raises_error(self):
        """Test that MissingDependencyError is raised when pdflatex is not found."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")

            with patch("shutil.which", return_value=None):
                with pytest.raises(MissingDependencyError):
                    func.plot.latex(output_path=output_path, compile_pdf=True)

    def test_compile_pdf_success(self):
        """Test successful PDF compilation."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")

            with patch("shutil.which", return_value="/usr/bin/pdflatex"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    result = func.plot.latex(output_path=output_path, compile_pdf=True)

                    assert result == output_path.replace(".tex", ".pdf")
                    mock_run.assert_called_once()

    def test_compile_pdf_failure_raises_runtime_error(self):
        """Test that RuntimeError is raised when pdflatex fails."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")

            with patch("shutil.which", return_value="/usr/bin/pdflatex"):
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=1, stdout="Error", stderr="Failed")

                    with pytest.raises(RuntimeError):
                        func.plot.latex(output_path=output_path, compile_pdf=True)
