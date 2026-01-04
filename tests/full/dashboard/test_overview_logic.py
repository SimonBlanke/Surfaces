# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for overview page logic.

These tests verify the business logic of the overview page
without running the Streamlit UI.
"""

import pytest

from tests.conftest import HAS_STREAMLIT, requires_streamlit

if HAS_STREAMLIT:
    from surfaces._surrogates._dashboard._pages.overview import (
        R2_THRESHOLD,
        get_status,
        get_status_color,
    )


# =============================================================================
# Status Logic Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestStatusLogic:
    """Tests for status determination logic."""

    def test_status_missing_no_surrogate(self):
        """Status is 'Missing' when no surrogate exists."""
        row = {"has_surrogate": False, "latest_r2": None}
        assert get_status(row) == "Missing"

    def test_status_not_validated(self):
        """Status is 'Not Validated' when surrogate exists but no R2."""
        row = {"has_surrogate": True, "latest_r2": None}
        assert get_status(row) == "Not Validated"

    def test_status_needs_attention_low_r2(self):
        """Status is 'Needs Attention' when R2 is below threshold."""
        row = {"has_surrogate": True, "latest_r2": R2_THRESHOLD - 0.01}
        assert get_status(row) == "Needs Attention"

    def test_status_good_high_r2(self):
        """Status is 'Good' when R2 meets or exceeds threshold."""
        row = {"has_surrogate": True, "latest_r2": R2_THRESHOLD}
        assert get_status(row) == "Good"

        row = {"has_surrogate": True, "latest_r2": 0.99}
        assert get_status(row) == "Good"

    def test_status_edge_case_zero_r2(self):
        """Status handles zero R2 correctly."""
        row = {"has_surrogate": True, "latest_r2": 0.0}
        assert get_status(row) == "Needs Attention"


# =============================================================================
# Status Color Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestStatusColors:
    """Tests for status color mapping."""

    def test_good_color(self):
        """Good status has green color."""
        color = get_status_color("Good")
        assert color == "#28a745"

    def test_needs_attention_color(self):
        """Needs Attention status has yellow/amber color."""
        color = get_status_color("Needs Attention")
        assert color == "#ffc107"

    def test_not_validated_color(self):
        """Not Validated status has blue/info color."""
        color = get_status_color("Not Validated")
        assert color == "#17a2b8"

    def test_missing_color(self):
        """Missing status has red color."""
        color = get_status_color("Missing")
        assert color == "#dc3545"

    def test_unknown_status_fallback(self):
        """Unknown status returns gray fallback."""
        color = get_status_color("Unknown")
        assert color == "#6c757d"


# =============================================================================
# Threshold Configuration Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestThresholdConfig:
    """Tests for threshold configuration."""

    def test_r2_threshold_is_high(self):
        """R2 threshold is set to a high value (0.95)."""
        assert R2_THRESHOLD >= 0.90
        assert R2_THRESHOLD <= 1.0

    def test_threshold_boundary_behavior(self):
        """Status changes exactly at threshold boundary."""
        just_below = {"has_surrogate": True, "latest_r2": R2_THRESHOLD - 0.001}
        exactly_at = {"has_surrogate": True, "latest_r2": R2_THRESHOLD}
        just_above = {"has_surrogate": True, "latest_r2": R2_THRESHOLD + 0.001}

        assert get_status(just_below) == "Needs Attention"
        assert get_status(exactly_at) == "Good"
        assert get_status(just_above) == "Good"
