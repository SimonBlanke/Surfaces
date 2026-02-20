# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for the plot compatibility system.

Tests the declarative requirements checking in _compatibility.py:
- PlotRequirements.check() with its three constraint types
- PLOT_REGISTRY consistency with PlotAccessor
- available_plots() filtering logic
"""

from surfaces._visualize._compatibility import (
    PLOT_REGISTRY,
    PlotRequirements,
    available_plots,
    check_compatibility,
)
from surfaces.test_functions.algebraic import (
    AckleyFunction,
    ForresterFunction,
    SphereFunction,
)

# =========================================================================
# PlotRequirements.check() -- dimension constraints
# =========================================================================


class TestExactDimensionConstraint:
    """Test that dimensions=int requires an exact match."""

    def test_matching_dimensions_pass(self):
        """A 2D function satisfies dimensions=2."""
        req = PlotRequirements("test", "desc", dimensions=2)
        ok, reason = req.check(AckleyFunction())
        assert ok is True
        assert reason is None

    def test_wrong_dimensions_rejected(self):
        """A 5D function is rejected by dimensions=2."""
        req = PlotRequirements("test", "desc", dimensions=2)
        ok, reason = req.check(SphereFunction(n_dim=5))
        assert ok is False
        assert "5D" in reason

    def test_1d_function_rejected_by_exact_2(self):
        """A 1D function is rejected by dimensions=2."""
        req = PlotRequirements("test", "desc", dimensions=2)
        ok, reason = req.check(ForresterFunction())
        assert ok is False


class TestRangeDimensionConstraint:
    """Test that dimensions=(min, max) enforces a range."""

    def test_within_range_passes(self):
        """A 3D function satisfies dimensions=(2, 5)."""
        req = PlotRequirements("test", "desc", dimensions=(2, 5))
        ok, _ = req.check(SphereFunction(n_dim=3))
        assert ok is True

    def test_below_minimum_rejected(self):
        """A 1D function is rejected by dimensions=(2, None)."""
        req = PlotRequirements("test", "desc", dimensions=(2, None))
        ok, reason = req.check(ForresterFunction())
        assert ok is False
        assert "at least 2D" in reason

    def test_above_maximum_rejected(self):
        """A 5D function is rejected by dimensions=(None, 3)."""
        req = PlotRequirements("test", "desc", dimensions=(None, 3))
        ok, reason = req.check(SphereFunction(n_dim=5))
        assert ok is False
        assert "at most 3D" in reason

    def test_unbounded_range_accepts_any(self):
        """dimensions=(1, None) accepts 1D through high-D."""
        req = PlotRequirements("test", "desc", dimensions=(1, None))
        assert req.check(ForresterFunction())[0] is True
        assert req.check(SphereFunction(n_dim=50))[0] is True


# =========================================================================
# PlotRequirements.check() -- history constraint
# =========================================================================


class TestHistoryConstraint:
    """Test that requires_history gates on history availability."""

    def test_rejected_without_history(self):
        """requires_history=True rejects when no history available."""
        req = PlotRequirements("test", "desc", dimensions=(1, None), requires_history=True)
        ok, reason = req.check(AckleyFunction(), has_history=False)
        assert ok is False
        assert "history" in reason

    def test_passes_with_history(self):
        """requires_history=True passes when history is available."""
        req = PlotRequirements("test", "desc", dimensions=(1, None), requires_history=True)
        ok, _ = req.check(AckleyFunction(), has_history=True)
        assert ok is True


# =========================================================================
# PlotRequirements.check() -- attribute constraint
# =========================================================================


class TestAttributeConstraint:
    """Test that requires_attribute gates on function attributes."""

    def test_rejected_without_attribute(self):
        """Function without the required attribute is rejected."""
        req = PlotRequirements(
            "test", "desc", dimensions=(1, None), requires_attribute="nonexistent_attr"
        )
        ok, reason = req.check(AckleyFunction())
        assert ok is False
        assert "nonexistent_attr" in reason

    def test_passes_with_attribute(self):
        """Function with the required attribute passes."""
        req = PlotRequirements("test", "desc", dimensions=2, requires_attribute="latex_formula")
        # AckleyFunction has latex_formula
        ok, _ = req.check(AckleyFunction())
        assert ok is True

    def test_no_attribute_requirement_always_passes(self):
        """requires_attribute=None imposes no constraint."""
        req = PlotRequirements("test", "desc", dimensions=2, requires_attribute=None)
        ok, _ = req.check(AckleyFunction())
        assert ok is True


# =========================================================================
# PlotRequirements.check() -- evaluation order
# =========================================================================


class TestCheckEvaluationOrder:
    """Test that check() short-circuits: dimensions checked before attribute.

    This matters because a 5D function should get a dimension error,
    not an attribute error, even if both would fail.
    """

    def test_dimension_failure_takes_priority_over_attribute(self):
        """Dimension mismatch reported even when attribute also missing."""
        req = PlotRequirements("test", "desc", dimensions=2, requires_attribute="nonexistent_attr")
        ok, reason = req.check(SphereFunction(n_dim=5))
        assert ok is False
        assert "5D" in reason  # dimension error, not attribute error


# =========================================================================
# PLOT_REGISTRY consistency
# =========================================================================


class TestRegistryConsistency:
    """Test that PLOT_REGISTRY stays in sync with PlotAccessor."""

    def test_all_registry_entries_exist_on_accessor(self):
        """Every key in PLOT_REGISTRY must be a method on PlotAccessor."""
        from surfaces._visualize._accessor import PlotAccessor

        for plot_name in PLOT_REGISTRY:
            assert hasattr(
                PlotAccessor, plot_name
            ), f"PLOT_REGISTRY has '{plot_name}' but PlotAccessor has no such method"

    def test_latex_requires_latex_formula(self):
        """The latex entry must gate on the latex_formula attribute.

        Without this, available() would claim latex works for BBOB/simulation
        functions that have no formula, and func.plot.latex() would crash.
        """
        req = PLOT_REGISTRY["latex"]
        assert req.requires_attribute == "latex_formula"


# =========================================================================
# available_plots() -- integration
# =========================================================================


class TestAvailablePlots:
    """Test the available_plots() function that drives func.plot.available()."""

    def test_2d_algebraic_includes_surface_and_latex(self):
        """2D algebraic function with formula gets surface + latex."""
        func = AckleyFunction()
        names = [p["name"] for p in available_plots(func)]
        assert "surface" in names
        assert "latex" in names

    def test_nd_algebraic_excludes_2d_plots(self):
        """5D function must not include surface/contour/heatmap."""
        func = SphereFunction(n_dim=5)
        names = [p["name"] for p in available_plots(func)]
        assert "surface" not in names
        assert "contour" not in names
        assert "heatmap" not in names
        assert "multi_slice" in names

    def test_convergence_excluded_without_history(self):
        """Convergence requires history; excluded by default."""
        func = AckleyFunction()
        names = [p["name"] for p in available_plots(func, has_history=False)]
        assert "convergence" not in names

    def test_convergence_included_with_history(self):
        """Convergence appears when history is available."""
        func = AckleyFunction()
        names = [p["name"] for p in available_plots(func, has_history=True)]
        assert "convergence" in names

    def test_latex_excluded_for_function_without_formula(self):
        """Non-algebraic 2D function without latex_formula must not get latex."""
        from surfaces.test_functions.simulation import DampedOscillatorFunction

        func = DampedOscillatorFunction()  # 2D but no latex_formula
        names = [p["name"] for p in available_plots(func)]
        assert "latex" not in names
        # But surface/contour should still work
        assert "surface" in names


# =========================================================================
# check_compatibility() -- single-plot query
# =========================================================================


class TestCheckCompatibility:
    """Test the single-plot compatibility query."""

    def test_unknown_plot_name_rejected(self):
        """Unknown plot type returns (False, reason)."""
        ok, reason = check_compatibility(AckleyFunction(), "nonexistent_plot")
        assert ok is False
        assert "unknown" in reason

    def test_known_compatible_plot(self):
        """Known compatible plot returns (True, None)."""
        ok, reason = check_compatibility(AckleyFunction(), "surface")
        assert ok is True
        assert reason is None

    def test_known_incompatible_plot(self):
        """Known incompatible plot returns (False, reason)."""
        ok, reason = check_compatibility(SphereFunction(n_dim=5), "surface")
        assert ok is False
        assert reason is not None
