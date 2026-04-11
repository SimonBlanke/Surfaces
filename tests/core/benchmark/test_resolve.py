"""Tests for optimizer and function resolution via duck typing."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from surfaces.benchmark._resolve import (
    _get_module,
    _has_ask_tell,
    resolve_functions,
    resolve_optimizer,
)

# Fake optimizer classes that simulate different packages.
# We set __module__ to control which adapter gets matched.


def _make_class(module_path: str, name: str = "FakeOptimizer", with_ask_tell: bool = False):
    """Create a class whose __module__ points to the given package."""
    attrs = {"__module__": module_path}
    if with_ask_tell:
        attrs["ask"] = lambda self: {}
        attrs["tell"] = lambda self, params, score: None
    cls = type(name, (), attrs)
    cls.__module__ = module_path
    return cls


class TestGetModule:
    def test_class(self):
        cls = _make_class("gradient_free_optimizers.optimizers")
        assert _get_module(cls) == "gradient_free_optimizers.optimizers"

    def test_instance(self):
        cls = _make_class("optuna.samplers")
        assert _get_module(cls()) == "optuna.samplers"

    def test_builtin(self):
        assert _get_module(int) == "builtins"

    def test_none_module(self):
        cls = type("NoModule", (), {})
        cls.__module__ = None
        assert _get_module(cls) == ""


class TestHasAskTell:
    def test_with_ask_tell(self):
        cls = _make_class("foo", with_ask_tell=True)
        assert _has_ask_tell(cls) is True

    def test_without_ask_tell(self):
        cls = _make_class("foo")
        assert _has_ask_tell(cls) is False

    def test_only_ask(self):
        cls = type("OnlyAsk", (), {"ask": lambda self: {}})
        assert _has_ask_tell(cls) is False

    def test_only_tell(self):
        cls = type("OnlyTell", (), {"tell": lambda self, p, s: None})
        assert _has_ask_tell(cls) is False

    def test_instance(self):
        cls = _make_class("foo", with_ask_tell=True)
        assert _has_ask_tell(cls()) is True

    def test_non_callable_attributes(self):
        """ask/tell exist but aren't callable."""
        cls = type("BadOptimizer", (), {"ask": 42, "tell": "nope"})
        assert _has_ask_tell(cls) is False


class TestPackagePrefixBoundaries:
    """Verify that prefix matching respects package boundaries.

    "cma" must match "cma" and "cma.evolution_strategy" but NOT "cmath".
    """

    @pytest.mark.parametrize(
        "module_path, should_match_cma",
        [
            ("cma", True),
            ("cma.evolution_strategy", True),
            ("cma.s", True),
            ("cmath", False),
            ("cmath_optimizer", False),
            ("cma_extended", False),
        ],
    )
    def test_cma_prefix(self, module_path, should_match_cma):
        cls = _make_class(module_path, with_ask_tell=True)

        fake_adapter = type(
            "FakeAdapter",
            (),
            {
                "is_sealed": False,
                "name": "FakeCMA",
            },
        )()

        fake_module = types.ModuleType("surfaces.benchmark._adapters._cma")
        fake_module.ADAPTER_CLASS = lambda obj, params: fake_adapter

        with patch("surfaces.benchmark._resolve.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = fake_module

            if should_match_cma:
                result = resolve_optimizer(cls)
                assert result.name == "FakeCMA"
                mock_importlib.import_module.assert_called_once_with(
                    "surfaces.benchmark._adapters._cma"
                )
            else:
                # Should fall through to generic (has ask/tell) or raise
                mock_importlib.import_module.reset_mock()
                # For non-matching modules with ask/tell, the generic adapter is used
                result = resolve_optimizer(cls)
                mock_importlib.import_module.assert_not_called()

    @pytest.mark.parametrize(
        "module_path, expected_adapter_module",
        [
            ("gradient_free_optimizers", "_gfo"),
            ("gradient_free_optimizers.optimizers.hill_climbing", "_gfo"),
            ("scipy", "_scipy"),
            ("scipy.optimize", "_scipy"),
            ("optuna", "_optuna"),
            ("optuna.samplers", "_optuna"),
            ("nevergrad", "_nevergrad"),
            ("nevergrad.optimizers", "_nevergrad"),
            ("bayes_opt", "_bayesopt"),
            ("bayes_opt.bayesian_optimization", "_bayesopt"),
            ("pymoo", "_pymoo"),
            ("pymoo.algorithms.soo", "_pymoo"),
            ("skopt", "_skopt"),
            ("skopt.optimizer", "_skopt"),
            ("smac", "_smac"),
            ("smac.facade", "_smac"),
            ("pyswarms", "_pyswarms"),
            ("pyswarms.single", "_pyswarms"),
        ],
    )
    def test_all_registry_entries(self, module_path, expected_adapter_module):
        """Every registered package resolves to the correct adapter module."""
        cls = _make_class(module_path)

        fake_adapter = type("A", (), {"is_sealed": False, "name": "Resolved"})()
        fake_module = types.ModuleType("fake")
        fake_module.ADAPTER_CLASS = lambda obj, params: fake_adapter

        with patch("surfaces.benchmark._resolve.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = fake_module
            resolve_optimizer(cls)
            mock_importlib.import_module.assert_called_once_with(
                f"surfaces.benchmark._adapters.{expected_adapter_module}"
            )


class TestGenericFallback:
    def test_unknown_package_with_ask_tell(self):
        """Optimizer from an unknown package with ask/tell uses generic adapter."""
        cls = _make_class("my_custom_optimizer.core", with_ask_tell=True)
        adapter = resolve_optimizer(cls)
        assert adapter is not None

    def test_unknown_package_without_ask_tell_raises(self):
        cls = _make_class("totally_unknown.module")
        with pytest.raises(TypeError, match="Cannot resolve optimizer"):
            resolve_optimizer(cls)


class TestResolveOptimizerSpecs:
    def test_tuple_spec(self):
        """(class, params_dict) tuple is accepted."""
        cls = _make_class("gradient_free_optimizers.optimizers")

        fake_adapter = type("A", (), {"is_sealed": True, "name": "GFO"})()
        fake_module = types.ModuleType("fake")
        fake_module.ADAPTER_CLASS = lambda obj, params: fake_adapter

        with patch("surfaces.benchmark._resolve.importlib") as mock_importlib:
            mock_importlib.import_module.return_value = fake_module
            result = resolve_optimizer((cls, {"n_iter": 100}))
            assert result.name == "GFO"

    def test_bad_tuple_length(self):
        with pytest.raises(TypeError, match="must be.*class, params_dict"):
            resolve_optimizer(("a", "b", "c"))

    def test_bad_tuple_params_type(self):
        cls = _make_class("foo")
        with pytest.raises(TypeError, match="must be a dict"):
            resolve_optimizer((cls, [1, 2, 3]))


class TestResolveFunctions:
    def test_single_class(self):
        from surfaces.test_functions.algebraic import SphereFunction

        result = resolve_functions(SphereFunction)
        assert result == [SphereFunction]

    def test_list_of_classes(self):
        from surfaces.test_functions.algebraic import AckleyFunction, SphereFunction

        result = resolve_functions([SphereFunction, AckleyFunction])
        assert result == [SphereFunction, AckleyFunction]

    def test_collection(self):
        from surfaces import collection

        result = resolve_functions(collection.filter(category="algebraic", n_dim=2, unimodal=True))
        assert len(result) > 0
        assert all(isinstance(r, type) for r in result)
