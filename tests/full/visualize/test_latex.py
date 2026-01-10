# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for latex plot via PlotAccessor.

These tests verify the latex functionality through the accessor pattern.
"""

import os
import tempfile

import pytest

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    RastriginFunction,
    RosenbrockFunction,
    SphereFunction,
)
from surfaces.visualize._errors import PlotCompatibilityError


class TestLatexBasic:
    """Basic functionality tests for latex."""

    def test_ackley_function_generates_tex(self):
        """Test latex generation with AckleyFunction."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "ackley.tex")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path
            assert os.path.exists(output_path)

            # Check content
            with open(output_path) as f:
                content = f.read()
            assert "documentclass" in content
            assert "pgfplots" in content
            assert "Ackley" in content

    def test_rastrigin_function_generates_tex(self):
        """Test latex generation with RastriginFunction."""
        func = RastriginFunction(n_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "rastrigin.tex")
            result = func.plot.latex(output_path=output_path)

            assert os.path.exists(output_path)

    def test_rosenbrock_function_generates_tex(self):
        """Test latex generation with RosenbrockFunction."""
        func = RosenbrockFunction(n_dim=2)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "rosenbrock.tex")
            result = func.plot.latex(output_path=output_path)

            assert os.path.exists(output_path)

    def test_default_output_path(self):
        """Test that default output path uses function name."""
        func = AckleyFunction()

        # Use temp directory to avoid polluting working dir
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                result = func.plot.latex()

                # Should create file with function name
                assert result.endswith(".tex")
                assert os.path.exists(result)
            finally:
                os.chdir(original_cwd)

    def test_returns_tex_path(self):
        """Test that method returns path to generated file."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            result = func.plot.latex(output_path=output_path)

            assert isinstance(result, str)
            assert result.endswith(".tex")


class TestLatexErrors:
    """Test error handling in latex."""

    def test_error_for_nd_function(self):
        """Test that N-dimensional function raises error without params."""
        func = SphereFunction(n_dim=5)

        with pytest.raises(PlotCompatibilityError):
            func.plot.latex()

    def test_error_for_function_without_latex_formula(self):
        """Test that function without latex_formula raises error."""
        func = SphereFunction(n_dim=2)

        # SphereFunction should have latex_formula, but let's test the check
        # by temporarily removing it
        original = getattr(func, "latex_formula", None)
        if hasattr(func, "latex_formula"):
            delattr(func.__class__, "latex_formula")

        try:
            with pytest.raises(ValueError, match="latex_formula"):
                func.plot.latex()
        finally:
            # Restore
            if original is not None:
                func.__class__.latex_formula = original


class TestLatexContent:
    """Test latex file content."""

    def test_contains_formula(self):
        """Test that generated file contains the mathematical formula."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            # Should contain the latex formula
            assert "displaystyle" in content or "formula" in content.lower()

    def test_contains_title(self):
        """Test that generated file contains the function name as title."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            assert "Ackley" in content

    def test_custom_title(self):
        """Test that custom title is used."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, title="My Custom Title")

            with open(output_path) as f:
                content = f.read()

            assert "My Custom Title" in content

    def test_contains_pgfplots_surface(self):
        """Test that generated file contains pgfplots surface command."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            assert "addplot3" in content
            assert "surf" in content


class TestLatexParameters:
    """Test latex plot parameters."""

    def test_samples_parameter(self):
        """Test that samples parameter affects output."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, samples=50)

            with open(output_path) as f:
                content = f.read()

            assert "samples=50" in content

    def test_view_angles(self):
        """Test that view angle parameters are used."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(
                output_path=output_path,
                view_azimuth=-45,
                view_elevation=30,
            )

            with open(output_path) as f:
                content = f.read()

            assert "-45" in content
            assert "30" in content

    def test_zmin_zmax(self):
        """Test that z-axis limits are included when specified."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path, zmin=0, zmax=20)

            with open(output_path) as f:
                content = f.read()

            assert "zmin=0" in content
            assert "zmax=20" in content


class TestLatexWithParams:
    """Test latex with custom params."""

    def test_with_custom_bounds(self):
        """Test latex with custom bounds via params."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(
                output_path=output_path,
                params={"x0": (-2, 2), "x1": (-2, 2)},
            )

            with open(output_path) as f:
                content = f.read()

            # Bounds should be reflected in domain
            assert "domain=-2:2" in content


class TestLatexGlobalMinimum:
    """Test global minimum display in latex."""

    def test_shows_global_minimum(self):
        """Test that global minimum is shown for functions that have it."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.tex")
            func.plot.latex(output_path=output_path)

            with open(output_path) as f:
                content = f.read()

            # Should contain global minimum info
            assert "Global minimum" in content or "f_global" in str(func.f_global)


class TestLatexFileExtension:
    """Test file extension handling."""

    def test_adds_tex_extension(self):
        """Test that .tex extension is added if missing."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "myfile")  # No extension
            result = func.plot.latex(output_path=output_path)

            assert result.endswith(".tex")
            assert os.path.exists(result)

    def test_keeps_tex_extension(self):
        """Test that existing .tex extension is kept."""
        func = AckleyFunction()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "myfile.tex")
            result = func.plot.latex(output_path=output_path)

            assert result == output_path
            assert not result.endswith(".tex.tex")
