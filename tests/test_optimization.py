import numpy as np
import pytest
from gradient_free_optimizers import RandomSearchOptimizer

from surfaces.test_functions.algebraic import (
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    EasomFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    RastriginFunction,
    RosenbrockFunction,
    SimionescuFunction,
    SphereFunction,
    StyblinskiTangFunction,
)

sphere_function = SphereFunction(2)
rastrigin_function = RastriginFunction(2)
ackley_function = AckleyFunction()
rosenbrock_function = RosenbrockFunction(2)
beale_function = BealeFunction()
himmelblaus_function = HimmelblausFunction()
hölder_table_function = HölderTableFunction()
cross_in_tray_function = CrossInTrayFunction()
simionescu_function = SimionescuFunction()
easom_function = EasomFunction()
booth_function = BoothFunction()
goldstein_price_function = GoldsteinPriceFunction()
styblinski_tang_function = StyblinskiTangFunction(2)
bukin_function_n6 = BukinFunctionN6()


objective_function_para_2D = (
    "test_function",
    [
        (sphere_function),
        (rastrigin_function),
        (ackley_function),
        (rosenbrock_function),
        (beale_function),
        (himmelblaus_function),
        (hölder_table_function),
        (cross_in_tray_function),
        (simionescu_function),
        (easom_function),
        (booth_function),
        (goldstein_price_function),
        (styblinski_tang_function),
        (bukin_function_n6),
    ],
)


@pytest.mark.parametrize(*objective_function_para_2D)
def test_optimization_2D(test_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)


############################################################

sphere_function = SphereFunction(3)
rastrigin_function = RastriginFunction(3)
rosenbrock_function = RosenbrockFunction(3)


objective_function_para_3D = (
    "test_function",
    [
        (sphere_function),
        (rastrigin_function),
        (rosenbrock_function),
    ],
)


@pytest.mark.parametrize(*objective_function_para_3D)
def test_optimization_3D(test_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)


############################################################

sphere_function = SphereFunction(4)
rastrigin_function = RastriginFunction(4)
rosenbrock_function = RosenbrockFunction(4)


objective_function_para_4D = (
    "test_function",
    [
        (sphere_function),
        (rastrigin_function),
        (rosenbrock_function),
    ],
)


@pytest.mark.parametrize(*objective_function_para_4D)
def test_optimization_4D(test_function):
    search_space = {
        "x0": np.arange(0, 100, 1),
        "x1": np.arange(0, 100, 1),
        "x2": np.arange(0, 100, 1),
        "x3": np.arange(0, 100, 1),
    }

    opt = RandomSearchOptimizer(search_space)
    opt.search(test_function, n_iter=30)


############################################################
# scipy integration test


def test_scipy_integration():
    """Test that Surfaces functions work directly with scipy.optimize."""
    from scipy.optimize import minimize

    func = SphereFunction(n_dim=2)

    # func accepts numpy arrays directly
    result = minimize(
        func,
        x0=[1.0, 1.0],
        bounds=[(-5, 5), (-5, 5)],
        method="L-BFGS-B",
    )

    # Should find minimum near [0, 0]
    assert result.success
    assert abs(result.fun) < 0.01
    assert np.allclose(result.x, [0, 0], atol=0.1)


############################################################
# Optuna integration test


def test_optuna_integration():
    """Test that Surfaces functions work with Optuna."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = SphereFunction(n_dim=2)

    def objective(trial):
        x0 = trial.suggest_float("x0", -5, 5)
        x1 = trial.suggest_float("x1", -5, 5)
        # Surfaces accepts dict input
        return func({"x0": x0, "x1": x1})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    # Should find minimum near [0, 0]
    assert study.best_value < 0.5
    assert abs(study.best_params["x0"]) < 1.0
    assert abs(study.best_params["x1"]) < 1.0


def test_optuna_integration_rastrigin():
    """Test Optuna with a more complex function."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = RastriginFunction(n_dim=2)

    def objective(trial):
        params = {f"x{i}": trial.suggest_float(f"x{i}", -5, 5) for i in range(2)}
        return func(params)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, show_progress_bar=False)

    # Rastrigin has global minimum at origin with value 0
    # With 100 trials, we should get reasonably close
    assert study.best_value < 5.0


############################################################
# scikit-optimize integration test


def test_skopt_integration():
    """Test that Surfaces functions work with scikit-optimize."""
    pytest.importorskip("skopt")
    from skopt import gp_minimize

    func = SphereFunction(n_dim=2)

    # skopt passes list of values
    def objective(x):
        return func(x)

    result = gp_minimize(
        objective,
        [(-5.0, 5.0), (-5.0, 5.0)],
        n_calls=30,
        random_state=42,
    )

    # Should find minimum near [0, 0]
    assert result.fun < 0.5
    assert abs(result.x[0]) < 1.0
    assert abs(result.x[1]) < 1.0
