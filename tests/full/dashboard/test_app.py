# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for Streamlit dashboard app using AppTest.

These tests run the dashboard in headless mode without a browser.
Streamlit's AppTest allows testing UI components programmatically.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import HAS_STREAMLIT, requires_streamlit

if HAS_STREAMLIT:
    from streamlit.testing.v1 import AppTest

    from surfaces._surrogates._dashboard.database import init_db


@pytest.fixture
def mock_db():
    """Create a temporary database and mock the DB_PATH."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    db_path.unlink(missing_ok=True)


# =============================================================================
# App Loading Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestAppLoading:
    """Tests for basic app loading."""

    def test_app_runs_without_error(self, mock_db):
        """App initializes and runs without exceptions."""
        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 0},
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=10,
            )
            at.run()

            assert not at.exception, f"App raised exception: {at.exception}"

    def test_app_has_title(self, mock_db):
        """App displays the dashboard title."""
        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 0},
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=10,
            )
            at.run()

            assert len(at.title) > 0
            assert "Surrogate Dashboard" in at.title[0].value


# =============================================================================
# Sidebar Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestSidebar:
    """Tests for sidebar components."""

    def test_sidebar_has_metrics(self, mock_db):
        """Sidebar displays quick stats metrics."""
        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 0},
        ), patch(
            "surfaces._surrogates._dashboard.app.get_dashboard_stats",
            return_value={
                "total_functions": 10,
                "with_surrogate": 7,
                "without_surrogate": 3,
                "total_validations": 20,
                "total_trainings": 5,
            },
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=10,
            )
            at.run()

            # Check that metrics are rendered
            assert len(at.metric) >= 4

    def test_sidebar_has_sync_button(self, mock_db):
        """Sidebar has a sync database button."""
        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 0},
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=10,
            )
            at.run()

            # Find the sync button
            buttons = [b for b in at.button if "Sync" in str(b.label)]
            assert len(buttons) > 0


# =============================================================================
# Tab Navigation Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
class TestTabNavigation:
    """Tests for tab-based navigation."""

    def test_app_has_tabs(self, mock_db):
        """App displays the four main tabs."""
        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 0},
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=10,
            )
            at.run()

            # Check that tabs exist
            assert len(at.tabs) > 0


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.dashboard
@requires_streamlit
@pytest.mark.slow
class TestIntegration:
    """Integration tests with actual database operations."""

    def test_full_workflow(self, mock_db):
        """Test complete workflow: load, display stats, interact."""
        from surfaces._surrogates._dashboard.database import (
            insert_validation_run,
            upsert_surrogate,
        )

        # Populate test data
        upsert_surrogate(
            "TestClassifier",
            "classification",
            True,
            metadata={"n_samples": 1000, "training_r2": 0.96},
            db_path=mock_db,
        )
        insert_validation_run(
            "TestClassifier",
            "random",
            100,
            {"r2": 0.95, "mae": 0.01},
            {"avg_real_ms": 10, "avg_surrogate_ms": 0.1},
            db_path=mock_db,
        )

        with patch("surfaces._surrogates._dashboard.database.DB_PATH", mock_db), patch(
            "surfaces._surrogates._dashboard.app.init_db"
        ), patch(
            "surfaces._surrogates._dashboard.app.sync_all",
            return_value={"synced": 1},
        ):
            at = AppTest.from_file(
                "src/surfaces/_surrogates/_dashboard/app.py",
                default_timeout=15,
            )
            at.run()

            # Verify app loaded without errors
            assert not at.exception

            # Stats should reflect the test data
            stats_metrics = at.metric
            assert len(stats_metrics) >= 1
