# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for dashboard database operations.

These tests verify database CRUD operations without running the UI.
Uses a temporary database to avoid affecting production data.
"""

import tempfile
from pathlib import Path

import pytest

from surfaces._surrogates._dashboard.database import (
    get_all_surrogates,
    get_connection,
    get_dashboard_stats,
    get_functions_needing_training,
    get_latest_validation,
    get_overview_data,
    get_surrogate,
    get_training_jobs,
    get_validation_runs,
    init_db,
    insert_training_job,
    insert_validation_run,
    update_training_job,
    upsert_surrogate,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    init_db(db_path)
    yield db_path
    db_path.unlink(missing_ok=True)


# =============================================================================
# Database Initialization Tests
# =============================================================================


@pytest.mark.dashboard
class TestDatabaseInit:
    """Tests for database initialization."""

    def test_init_creates_tables(self, temp_db):
        """Database initialization creates all required tables."""
        with get_connection(temp_db) as conn:
            # Check tables exist
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            table_names = [t[0] for t in tables]

            assert "surrogates" in table_names
            assert "validation_runs" in table_names
            assert "training_jobs" in table_names

    def test_init_is_idempotent(self, temp_db):
        """Calling init_db multiple times doesn't cause errors."""
        init_db(temp_db)
        init_db(temp_db)

        with get_connection(temp_db) as conn:
            count = conn.execute("SELECT COUNT(*) FROM surrogates").fetchone()[0]
            assert count == 0


# =============================================================================
# Surrogate CRUD Tests
# =============================================================================


@pytest.mark.dashboard
class TestSurrogateCRUD:
    """Tests for surrogate record operations."""

    def test_upsert_creates_record(self, temp_db):
        """Upsert creates a new surrogate record."""
        upsert_surrogate(
            function_name="TestFunction",
            function_type="classification",
            has_surrogate=True,
            db_path=temp_db,
        )

        result = get_surrogate("TestFunction", db_path=temp_db)

        assert result is not None
        assert result["function_name"] == "TestFunction"
        assert result["function_type"] == "classification"
        assert result["has_surrogate"] == 1

    def test_upsert_updates_existing(self, temp_db):
        """Upsert updates an existing record."""
        upsert_surrogate(
            function_name="TestFunction",
            function_type="classification",
            has_surrogate=False,
            db_path=temp_db,
        )

        upsert_surrogate(
            function_name="TestFunction",
            function_type="classification",
            has_surrogate=True,
            db_path=temp_db,
        )

        result = get_surrogate("TestFunction", db_path=temp_db)
        assert result["has_surrogate"] == 1

    def test_upsert_with_metadata(self, temp_db):
        """Upsert stores metadata correctly."""
        metadata = {
            "param_names": ["alpha", "beta"],
            "param_encodings": {"alpha": "linear", "beta": "log"},
            "n_samples": 1000,
            "training_r2": 0.95,
            "training_mse": 0.001,
            "y_range": [0.0, 1.0],
        }

        upsert_surrogate(
            function_name="TestFunction",
            function_type="regression",
            has_surrogate=True,
            metadata=metadata,
            db_path=temp_db,
        )

        result = get_surrogate("TestFunction", db_path=temp_db)

        assert result["param_names"] == ["alpha", "beta"]
        assert result["param_encodings"] == {"alpha": "linear", "beta": "log"}
        assert result["n_samples"] == 1000
        assert result["training_r2"] == 0.95

    def test_get_nonexistent_surrogate(self, temp_db):
        """Getting a non-existent surrogate returns None."""
        result = get_surrogate("NonExistent", db_path=temp_db)
        assert result is None

    def test_get_all_surrogates(self, temp_db):
        """Get all surrogates returns all records."""
        upsert_surrogate("Func1", "classification", True, db_path=temp_db)
        upsert_surrogate("Func2", "regression", False, db_path=temp_db)
        upsert_surrogate("Func3", "classification", True, db_path=temp_db)

        results = get_all_surrogates(db_path=temp_db)

        assert len(results) == 3
        names = [r["function_name"] for r in results]
        assert "Func1" in names
        assert "Func2" in names
        assert "Func3" in names


# =============================================================================
# Validation Runs Tests
# =============================================================================


@pytest.mark.dashboard
class TestValidationRuns:
    """Tests for validation run operations."""

    def test_insert_validation_run(self, temp_db):
        """Insert validation run creates a record."""
        upsert_surrogate("TestFunc", "classification", True, db_path=temp_db)

        run_id = insert_validation_run(
            function_name="TestFunc",
            validation_type="random",
            n_samples=100,
            metrics={"r2": 0.95, "mae": 0.01, "rmse": 0.02},
            timing={"avg_real_ms": 10.0, "avg_surrogate_ms": 0.1, "speedup": 100.0},
            random_seed=42,
            db_path=temp_db,
        )

        assert run_id is not None
        assert run_id > 0

    def test_get_validation_runs(self, temp_db):
        """Get validation runs returns all runs for a function."""
        upsert_surrogate("TestFunc", "classification", True, db_path=temp_db)

        for i in range(3):
            insert_validation_run(
                function_name="TestFunc",
                validation_type="random",
                n_samples=100 * (i + 1),
                metrics={"r2": 0.9 + i * 0.01},
                timing={},
                db_path=temp_db,
            )

        runs = get_validation_runs("TestFunc", db_path=temp_db)

        assert len(runs) == 3

    def test_get_latest_validation(self, temp_db):
        """Get latest validation returns a validation run."""
        upsert_surrogate("TestFunc", "classification", True, db_path=temp_db)

        insert_validation_run(
            function_name="TestFunc",
            validation_type="random",
            n_samples=100,
            metrics={"r2": 0.90},
            timing={},
            db_path=temp_db,
        )

        latest = get_latest_validation("TestFunc", db_path=temp_db)

        assert latest is not None
        assert latest["function_name"] == "TestFunc"
        assert latest["n_samples"] == 100
        assert latest["r2_score"] == 0.90

    def test_get_latest_validation_no_runs(self, temp_db):
        """Get latest validation returns None when no runs exist."""
        result = get_latest_validation("NonExistent", db_path=temp_db)
        assert result is None


# =============================================================================
# Training Jobs Tests
# =============================================================================


@pytest.mark.dashboard
class TestTrainingJobs:
    """Tests for training job operations."""

    def test_insert_training_job(self, temp_db):
        """Insert training job creates a running job."""
        job_id = insert_training_job(
            function_name="TestFunc",
            triggered_by="manual",
            db_path=temp_db,
        )

        jobs = get_training_jobs(function_name="TestFunc", db_path=temp_db)

        assert len(jobs) == 1
        assert jobs[0]["status"] == "running"
        assert jobs[0]["triggered_by"] == "manual"

    def test_update_training_job_completed(self, temp_db):
        """Update training job marks it as completed."""
        job_id = insert_training_job("TestFunc", "manual", db_path=temp_db)

        update_training_job(job_id, status="completed", db_path=temp_db)

        jobs = get_training_jobs(function_name="TestFunc", db_path=temp_db)
        assert jobs[0]["status"] == "completed"
        assert jobs[0]["completed_at"] is not None

    def test_update_training_job_failed(self, temp_db):
        """Update training job can mark it as failed with error."""
        job_id = insert_training_job("TestFunc", "manual", db_path=temp_db)

        update_training_job(
            job_id,
            status="failed",
            error_message="Out of memory",
            db_path=temp_db,
        )

        jobs = get_training_jobs(function_name="TestFunc", db_path=temp_db)
        assert jobs[0]["status"] == "failed"
        assert jobs[0]["error_message"] == "Out of memory"

    def test_get_training_jobs_by_status(self, temp_db):
        """Get training jobs can filter by status."""
        job1 = insert_training_job("Func1", "manual", db_path=temp_db)
        job2 = insert_training_job("Func2", "manual", db_path=temp_db)
        update_training_job(job1, "completed", db_path=temp_db)

        running = get_training_jobs(status="running", db_path=temp_db)
        completed = get_training_jobs(status="completed", db_path=temp_db)

        assert len(running) == 1
        assert len(completed) == 1


# =============================================================================
# Dashboard Query Tests
# =============================================================================


@pytest.mark.dashboard
class TestDashboardQueries:
    """Tests for dashboard-specific queries."""

    def test_get_overview_data(self, temp_db):
        """Get overview data joins surrogates with validation."""
        upsert_surrogate("Func1", "classification", True, db_path=temp_db)
        upsert_surrogate("Func2", "regression", False, db_path=temp_db)

        insert_validation_run(
            function_name="Func1",
            validation_type="random",
            n_samples=100,
            metrics={"r2": 0.95},
            timing={},
            db_path=temp_db,
        )

        data = get_overview_data(db_path=temp_db)

        assert len(data) == 2

        func1_data = next(d for d in data if d["function_name"] == "Func1")
        func2_data = next(d for d in data if d["function_name"] == "Func2")

        assert func1_data["latest_r2"] == 0.95
        assert func2_data["latest_r2"] is None

    def test_get_functions_needing_training(self, temp_db):
        """Get functions needing training identifies missing/low accuracy."""
        # Function without surrogate
        upsert_surrogate("Missing", "classification", False, db_path=temp_db)

        # Function with good surrogate
        upsert_surrogate("Good", "classification", True, db_path=temp_db)
        insert_validation_run("Good", "random", 100, {"r2": 0.98}, {}, db_path=temp_db)

        # Function with low accuracy
        upsert_surrogate("LowAccuracy", "classification", True, db_path=temp_db)
        insert_validation_run("LowAccuracy", "random", 100, {"r2": 0.80}, {}, db_path=temp_db)

        needs_training = get_functions_needing_training(r2_threshold=0.95, db_path=temp_db)

        assert "Missing" in needs_training
        assert "LowAccuracy" in needs_training
        assert "Good" not in needs_training

    def test_get_dashboard_stats(self, temp_db):
        """Get dashboard stats returns correct counts."""
        upsert_surrogate("Func1", "classification", True, db_path=temp_db)
        upsert_surrogate("Func2", "regression", True, db_path=temp_db)
        upsert_surrogate("Func3", "classification", False, db_path=temp_db)

        insert_validation_run("Func1", "random", 100, {}, {}, db_path=temp_db)
        insert_validation_run("Func1", "grid", 200, {}, {}, db_path=temp_db)

        job_id = insert_training_job("Func1", "manual", db_path=temp_db)
        update_training_job(job_id, "completed", db_path=temp_db)

        stats = get_dashboard_stats(db_path=temp_db)

        assert stats["total_functions"] == 3
        assert stats["with_surrogate"] == 2
        assert stats["without_surrogate"] == 1
        assert stats["total_validations"] == 2
        assert stats["total_trainings"] == 1
