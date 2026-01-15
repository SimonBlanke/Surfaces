# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Tests for CustomTestFunction storage backends."""

from __future__ import annotations

import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from surfaces.custom_test_function import (
    CustomTestFunction,
    InMemoryStorage,
    SQLiteStorage,
    Storage,
)

# =============================================================================
# Test Fixtures
# =============================================================================


def sphere(params: Dict[str, Any]) -> float:
    """Simple sphere function for testing."""
    return sum(v**2 for v in params.values())


@pytest.fixture
def search_space():
    """Standard 2D search space."""
    return {"x": (-5.0, 5.0), "y": (-5.0, 5.0)}


@pytest.fixture
def temp_db_path():
    """Temporary database file path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    yield path
    # Cleanup
    if path.exists():
        path.unlink()


# =============================================================================
# InMemoryStorage Tests
# =============================================================================


class TestInMemoryStorage:
    """Tests for InMemoryStorage backend."""

    def test_init_default_experiment(self):
        """InMemoryStorage uses 'default' as default experiment name."""
        storage = InMemoryStorage()
        assert storage.experiment == "default"

    def test_init_custom_experiment(self):
        """InMemoryStorage accepts custom experiment name."""
        storage = InMemoryStorage(experiment="my-exp")
        assert storage.experiment == "my-exp"

    def test_save_and_load_evaluation(self):
        """Can save and load evaluations."""
        storage = InMemoryStorage()

        storage.save_evaluation({"x": 1.0, "y": 2.0, "score": 5.0})
        storage.save_evaluation({"x": 0.0, "y": 0.0, "score": 0.0})

        evaluations = storage.load_evaluations()
        assert len(evaluations) == 2
        assert evaluations[0]["x"] == 1.0
        assert evaluations[0]["score"] == 5.0
        assert evaluations[1]["score"] == 0.0

    def test_evaluation_metadata_added(self):
        """Evaluations get timestamp and ID added automatically."""
        storage = InMemoryStorage()
        storage.save_evaluation({"x": 1.0, "score": 1.0})

        evaluations = storage.load_evaluations()
        assert "_timestamp" in evaluations[0]
        assert "_evaluation_id" in evaluations[0]
        assert evaluations[0]["_evaluation_id"] == 1

    def test_evaluation_ids_increment(self):
        """Evaluation IDs increment properly."""
        storage = InMemoryStorage()

        for i in range(5):
            storage.save_evaluation({"x": float(i), "score": float(i)})

        evaluations = storage.load_evaluations()
        ids = [e["_evaluation_id"] for e in evaluations]
        assert ids == [1, 2, 3, 4, 5]

    def test_save_and_load_state(self):
        """Can save and load state/checkpoint."""
        storage = InMemoryStorage()

        state = {
            "n_evaluations": 10,
            "best_score": 0.5,
            "best_params": {"x": 0.1, "y": 0.2},
            "total_time": 5.5,
        }
        storage.save_state(state)

        loaded = storage.load_state()
        assert loaded is not None
        assert loaded["n_evaluations"] == 10
        assert loaded["best_score"] == 0.5
        assert "_checkpoint_timestamp" in loaded

    def test_load_state_returns_none_if_empty(self):
        """load_state returns None if no state saved."""
        storage = InMemoryStorage()
        assert storage.load_state() is None

    def test_delete_experiment(self):
        """delete_experiment clears all data."""
        storage = InMemoryStorage()

        storage.save_evaluation({"x": 1.0, "score": 1.0})
        storage.save_state({"n_evaluations": 1})

        storage.delete_experiment()

        assert storage.load_evaluations() == []
        assert storage.load_state() is None

    def test_close_is_noop(self):
        """close() works without error (no-op for memory)."""
        storage = InMemoryStorage()
        storage.save_evaluation({"x": 1.0, "score": 1.0})
        storage.close()
        # Should still work after close (no actual resources to release)
        assert len(storage.load_evaluations()) == 1

    def test_repr(self):
        """__repr__ shows useful info."""
        storage = InMemoryStorage(experiment="test")
        storage.save_evaluation({"x": 1.0, "score": 1.0})

        repr_str = repr(storage)
        assert "InMemoryStorage" in repr_str
        assert "test" in repr_str
        assert "1" in repr_str


# =============================================================================
# SQLiteStorage Tests
# =============================================================================


class TestSQLiteStorage:
    """Tests for SQLiteStorage backend."""

    def test_init_creates_database(self, temp_db_path):
        """SQLiteStorage creates database file."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        assert temp_db_path.exists()
        storage.close()

    def test_init_creates_parent_directory(self):
        """SQLiteStorage creates parent directory if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "subdir" / "experiments.db"
            storage = SQLiteStorage(db_path, experiment="test")
            assert db_path.parent.exists()
            storage.close()

    def test_memory_database(self):
        """SQLiteStorage supports :memory: for in-memory SQLite."""
        storage = SQLiteStorage(":memory:", experiment="test")
        storage.save_evaluation({"x": 1.0, "score": 1.0})
        assert len(storage.load_evaluations()) == 1
        storage.close()

    def test_save_and_load_evaluation(self, temp_db_path):
        """Can save and load evaluations."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        storage.save_evaluation({"x": 1.0, "y": 2.0, "score": 5.0})
        storage.save_evaluation({"x": 0.0, "y": 0.0, "score": 0.0})

        evaluations = storage.load_evaluations()
        assert len(evaluations) == 2
        assert evaluations[0]["x"] == 1.0
        assert evaluations[0]["score"] == 5.0
        storage.close()

    def test_evaluations_persist_across_connections(self, temp_db_path):
        """Evaluations persist when reopening database."""
        storage1 = SQLiteStorage(temp_db_path, experiment="test")
        storage1.save_evaluation({"x": 1.0, "score": 1.0})
        storage1.save_evaluation({"x": 2.0, "score": 4.0})
        storage1.close()

        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        evaluations = storage2.load_evaluations()
        assert len(evaluations) == 2
        storage2.close()

    def test_multiple_experiments_same_database(self, temp_db_path):
        """Multiple experiments can share same database file."""
        storage1 = SQLiteStorage(temp_db_path, experiment="exp-1")
        storage2 = SQLiteStorage(temp_db_path, experiment="exp-2")

        storage1.save_evaluation({"x": 1.0, "score": 1.0})
        storage1.save_evaluation({"x": 2.0, "score": 4.0})
        storage2.save_evaluation({"x": 10.0, "score": 100.0})

        assert len(storage1.load_evaluations()) == 2
        assert len(storage2.load_evaluations()) == 1

        storage1.close()
        storage2.close()

    def test_save_and_load_state(self, temp_db_path):
        """Can save and load state/checkpoint."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        state = {
            "n_evaluations": 10,
            "best_score": 0.5,
            "best_params": {"x": 0.1, "y": 0.2},
            "total_time": 5.5,
            "memory_cache": {(1.0, 2.0): 5.0, (0.0, 0.0): 0.0},
        }
        storage.save_state(state)

        loaded = storage.load_state()
        assert loaded is not None
        assert loaded["n_evaluations"] == 10
        assert loaded["best_score"] == 0.5
        assert loaded["memory_cache"] == {(1.0, 2.0): 5.0, (0.0, 0.0): 0.0}
        storage.close()

    def test_state_persists_across_connections(self, temp_db_path):
        """State persists when reopening database."""
        storage1 = SQLiteStorage(temp_db_path, experiment="test")
        storage1.save_state({"n_evaluations": 42, "total_time": 10.0})
        storage1.close()

        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        state = storage2.load_state()
        assert state is not None
        assert state["n_evaluations"] == 42
        storage2.close()

    def test_delete_experiment(self, temp_db_path):
        """delete_experiment removes only this experiment's data."""
        storage1 = SQLiteStorage(temp_db_path, experiment="exp-1")
        storage2 = SQLiteStorage(temp_db_path, experiment="exp-2")

        storage1.save_evaluation({"x": 1.0, "score": 1.0})
        storage2.save_evaluation({"x": 2.0, "score": 4.0})

        storage1.delete_experiment()

        assert len(storage1.load_evaluations()) == 0
        assert len(storage2.load_evaluations()) == 1

        storage1.close()
        storage2.close()

    def test_list_experiments(self, temp_db_path):
        """list_experiments returns all experiment names."""
        storage1 = SQLiteStorage(temp_db_path, experiment="exp-1")
        storage2 = SQLiteStorage(temp_db_path, experiment="exp-2")

        storage1.save_evaluation({"x": 1.0, "score": 1.0})
        storage2.save_evaluation({"x": 2.0, "score": 4.0})

        experiments = storage1.list_experiments()
        assert set(experiments) == {"exp-1", "exp-2"}

        storage1.close()
        storage2.close()

    def test_query_order_by_score(self, temp_db_path):
        """query() can order by score."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        storage.save_evaluation({"x": 1.0, "score": 5.0})
        storage.save_evaluation({"x": 2.0, "score": 1.0})
        storage.save_evaluation({"x": 3.0, "score": 10.0})

        # Ascending
        results = storage.query(order_by="score")
        scores = [r["score"] for r in results]
        assert scores == [1.0, 5.0, 10.0]

        # Descending
        results = storage.query(order_by="-score")
        scores = [r["score"] for r in results]
        assert scores == [10.0, 5.0, 1.0]

        storage.close()

    def test_query_with_limit(self, temp_db_path):
        """query() can limit results."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        for i in range(10):
            storage.save_evaluation({"x": float(i), "score": float(i)})

        results = storage.query(order_by="score", limit=3)
        assert len(results) == 3

        storage.close()

    def test_query_with_filter(self, temp_db_path):
        """query() can filter with Python function."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        for i in range(10):
            storage.save_evaluation({"x": float(i), "score": float(i)})

        results = storage.query(filter_fn=lambda e: e["x"] > 5)
        assert len(results) == 4
        assert all(r["x"] > 5 for r in results)

        storage.close()

    def test_thread_safety(self, temp_db_path):
        """SQLiteStorage is thread-safe."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(10):
                    storage.save_evaluation(
                        {
                            "thread": thread_id,
                            "i": i,
                            "score": float(thread_id * 100 + i),
                        }
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        evaluations = storage.load_evaluations()
        assert len(evaluations) == 50  # 5 threads * 10 evaluations

        # Ensure connection is fully closed and all threads are done
        storage.close()

        # Give Windows time to release file handles
        import time

        time.sleep(0.1)

    def test_context_manager(self, temp_db_path):
        """SQLiteStorage works as context manager."""
        with SQLiteStorage(temp_db_path, experiment="test") as storage:
            storage.save_evaluation({"x": 1.0, "score": 1.0})
            assert len(storage.load_evaluations()) == 1

        # Connection should be closed, but new one works
        with SQLiteStorage(temp_db_path, experiment="test") as storage:
            assert len(storage.load_evaluations()) == 1


# =============================================================================
# CustomTestFunction Storage Integration Tests
# =============================================================================


class TestCustomTestFunctionWithStorage:
    """Tests for CustomTestFunction with storage backends."""

    def test_without_storage_works(self, search_space):
        """CustomTestFunction works without storage (default)."""
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
        )

        result = func(x=1.0, y=2.0)
        assert result == 5.0
        assert func.n_evaluations == 1
        assert not func.storage.is_configured

    def test_with_memory_storage(self, search_space):
        """CustomTestFunction works with InMemoryStorage."""
        storage = InMemoryStorage(experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        func(x=1.0, y=2.0)
        func(x=0.0, y=0.0)

        assert func.storage.backend is storage
        assert len(storage.load_evaluations()) == 2

    def test_with_sqlite_storage(self, search_space, temp_db_path):
        """CustomTestFunction works with SQLiteStorage."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        func(x=1.0, y=2.0)
        func(x=0.0, y=0.0)

        assert len(storage.load_evaluations()) == 2
        func.close()

    def test_evaluations_persisted_automatically(self, search_space, temp_db_path):
        """Each evaluation is persisted to storage automatically."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        func(x=1.0, y=2.0)

        # Check immediately after evaluation
        evaluations = storage.load_evaluations()
        assert len(evaluations) == 1
        assert evaluations[0]["x"] == 1.0
        assert evaluations[0]["y"] == 2.0
        assert evaluations[0]["score"] == 5.0

        func.close()

    def test_resume_from_storage(self, search_space, temp_db_path):
        """CustomTestFunction resumes from existing storage data."""
        # First session
        storage1 = SQLiteStorage(temp_db_path, experiment="test")
        func1 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage1,
        )
        func1(x=1.0, y=2.0)
        func1(x=0.0, y=0.0)
        func1.close()

        # Second session - should resume
        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        func2 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage2,
        )

        assert func2.n_evaluations == 2
        assert len(func2.search_data) == 2
        assert func2.best_score == 0.0
        assert func2.best_params == {"x": 0.0, "y": 0.0}

        func2.close()

    def test_resume_false_starts_fresh(self, search_space, temp_db_path):
        """resume=False ignores existing storage data."""
        # First session
        storage1 = SQLiteStorage(temp_db_path, experiment="test")
        func1 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage1,
        )
        func1(x=1.0, y=2.0)
        func1.close()

        # Second session with resume=False
        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        func2 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage2,
            resume=False,
        )

        assert func2.n_evaluations == 0
        assert len(func2.search_data) == 0

        func2.close()

    def test_save_checkpoint(self, search_space, temp_db_path):
        """save_checkpoint saves state to storage."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
            memory=True,
        )

        func(x=1.0, y=2.0)
        func(x=0.0, y=0.0)
        func.storage.save_checkpoint()

        state = storage.load_state()
        assert state is not None
        assert state["n_evaluations"] == 2
        assert state["best_score"] == 0.0

        func.close()

    def test_save_checkpoint_without_storage_raises(self, search_space):
        """save_checkpoint raises if no storage configured."""
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
        )

        with pytest.raises(RuntimeError, match="No storage backend"):
            func.storage.save_checkpoint()

    def test_load_checkpoint(self, search_space, temp_db_path):
        """load_checkpoint manually reloads state."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
            memory=True,
        )

        # Manually save some state
        storage.save_state(
            {
                "total_time": 99.9,
                "memory_cache": {(1.0, 2.0): 5.0},
            }
        )

        # Load it
        result = func.storage.load_checkpoint()

        assert result is True
        assert func.total_time == 99.9
        assert func._memory_cache == {(1.0, 2.0): 5.0}

        func.close()

    def test_load_checkpoint_returns_false_if_none(self, search_space, temp_db_path):
        """load_checkpoint returns False if no checkpoint exists."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        result = func.storage.load_checkpoint()
        assert result is False

        func.close()

    def test_load_checkpoint_without_storage_raises(self, search_space):
        """load_checkpoint raises if no storage configured."""
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
        )

        with pytest.raises(RuntimeError, match="No storage backend"):
            func.storage.load_checkpoint()

    def test_delete_experiment(self, search_space, temp_db_path):
        """delete clears storage and resets function."""
        storage = SQLiteStorage(temp_db_path, experiment="test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        func(x=1.0, y=2.0)
        func.storage.save_checkpoint()

        func.storage.delete()

        assert func.n_evaluations == 0
        assert len(storage.load_evaluations()) == 0
        assert storage.load_state() is None

        func.close()

    def test_delete_experiment_without_storage_raises(self, search_space):
        """delete raises if no storage configured."""
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
        )

        with pytest.raises(RuntimeError, match="No storage backend"):
            func.storage.delete()

    def test_context_manager(self, search_space, temp_db_path):
        """CustomTestFunction works as context manager."""
        storage = SQLiteStorage(temp_db_path, experiment="test")

        with CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        ) as func:
            func(x=1.0, y=2.0)
            assert func.n_evaluations == 1

        # Storage should still be accessible with new connection
        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        assert len(storage2.load_evaluations()) == 1
        storage2.close()

    def test_memory_cache_persisted_in_checkpoint(self, search_space, temp_db_path):
        """Memory cache is saved and restored via checkpoint."""
        # First session with memory caching
        storage1 = SQLiteStorage(temp_db_path, experiment="test")
        func1 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage1,
            memory=True,
        )

        func1(x=1.0, y=2.0)
        func1(x=1.0, y=2.0)  # Should hit cache
        func1.storage.save_checkpoint()
        func1.close()

        # Second session - should restore cache
        storage2 = SQLiteStorage(temp_db_path, experiment="test")
        func2 = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage2,
            memory=True,
        )

        # Cache should be restored
        assert len(func2._memory_cache) == 1

        func2.close()


# =============================================================================
# Custom Storage Implementation Test
# =============================================================================


class TestCustomStorageImplementation:
    """Test that users can implement custom storage backends."""

    def test_custom_storage_implementation(self, search_space):
        """Users can implement custom storage backends."""

        class ListStorage(Storage):
            """Simple list-based storage for testing."""

            def __init__(self, experiment: str = "default"):
                self._experiment = experiment
                self.evaluations: List[Dict[str, Any]] = []
                self.state: Optional[Dict[str, Any]] = None

            @property
            def experiment(self) -> str:
                return self._experiment

            def save_evaluation(self, evaluation: Dict[str, Any]) -> None:
                self.evaluations.append(evaluation.copy())

            def load_evaluations(self) -> List[Dict[str, Any]]:
                return self.evaluations.copy()

            def save_state(self, state: Dict[str, Any]) -> None:
                self.state = state.copy()

            def load_state(self) -> Optional[Dict[str, Any]]:
                return self.state.copy() if self.state else None

            def delete_experiment(self) -> None:
                self.evaluations = []
                self.state = None

            def close(self) -> None:
                pass

        # Use custom storage with CustomTestFunction
        storage = ListStorage(experiment="custom-test")
        func = CustomTestFunction(
            objective_fn=sphere,
            search_space=search_space,
            storage=storage,
        )

        func(x=1.0, y=2.0)
        func(x=0.0, y=0.0)

        assert len(storage.evaluations) == 2
        assert storage.evaluations[0]["x"] == 1.0
        assert storage.evaluations[1]["score"] == 0.0
