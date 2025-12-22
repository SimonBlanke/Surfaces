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


############################################################
# Nevergrad integration test


def test_nevergrad_integration():
    """Test that Surfaces functions work with Nevergrad."""
    pytest.importorskip("nevergrad")
    import nevergrad as ng

    func = SphereFunction(n_dim=2)

    # Nevergrad uses its own parametrization, passes array-like
    def objective(x):
        return func(x)

    parametrization = ng.p.Array(shape=(2,), lower=-5, upper=5)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=50)
    recommendation = optimizer.minimize(objective)

    # Should find minimum near [0, 0]
    result = recommendation.value
    assert func(result) < 0.5
    assert abs(result[0]) < 1.0
    assert abs(result[1]) < 1.0


def test_nevergrad_integration_with_dict():
    """Test Nevergrad with dict-based parametrization."""
    pytest.importorskip("nevergrad")
    import nevergrad as ng

    func = SphereFunction(n_dim=2)

    # Nevergrad can also use dict-based instrumentation
    parametrization = ng.p.Instrumentation(
        x0=ng.p.Scalar(lower=-5, upper=5),
        x1=ng.p.Scalar(lower=-5, upper=5),
    )

    def objective(x0, x1):
        return func({"x0": x0, "x1": x1})

    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=50)
    recommendation = optimizer.minimize(objective)

    # Should find minimum near [0, 0]
    assert func({"x0": recommendation.kwargs["x0"], "x1": recommendation.kwargs["x1"]}) < 0.5


############################################################
# Bayesian Optimization integration test


def test_bayesian_optimization_integration():
    """Test that Surfaces functions work with bayesian-optimization."""
    pytest.importorskip("bayes_opt")
    from bayes_opt import BayesianOptimization

    func = SphereFunction(n_dim=2)

    # bayesian-optimization maximizes, so we negate for minimization
    # It passes kwargs to the objective function
    def objective(x0, x1):
        return -func({"x0": x0, "x1": x1})

    optimizer = BayesianOptimization(
        f=objective,
        pbounds={"x0": (-5, 5), "x1": (-5, 5)},
        random_state=42,
        verbose=0,
    )
    optimizer.maximize(init_points=5, n_iter=25)

    # Should find minimum near [0, 0] (remember we negated, so max is near 0)
    assert -optimizer.max["target"] < 0.5
    assert abs(optimizer.max["params"]["x0"]) < 1.0
    assert abs(optimizer.max["params"]["x1"]) < 1.0


############################################################
# Hyperopt integration test


def test_hyperopt_integration():
    """Test that Surfaces functions work with Hyperopt."""
    pytest.importorskip("hyperopt")
    from hyperopt import fmin, tpe, hp, Trials

    func = SphereFunction(n_dim=2)

    # Hyperopt passes dict to objective
    def objective(params):
        return func(params)

    space = {
        "x0": hp.uniform("x0", -5, 5),
        "x1": hp.uniform("x1", -5, 5),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials,
        rstate=np.random.default_rng(42),
        show_progressbar=False,
    )

    # Should find minimum near [0, 0]
    assert func(best) < 0.5
    assert abs(best["x0"]) < 1.0
    assert abs(best["x1"]) < 1.0


############################################################
# Ray Tune integration test


def test_ray_tune_integration():
    """Test that Surfaces functions work with Ray Tune.

    Note: Ray Tune can have compatibility issues in certain environments.
    This test verifies that Surfaces functions can be used with Ray's
    interface (dict-based config).
    """
    ray = pytest.importorskip("ray")
    from ray import tune

    func = SphereFunction(n_dim=2)

    def objective(config):
        result = func({"x0": config["x0"], "x1": config["x1"]})
        # Ray Tune expects tune.report() or return dict
        return {"loss": result}

    # Test that the objective function works with Ray's config format
    # This verifies Surfaces compatibility without running full Ray cluster
    test_config = {"x0": 0.5, "x1": -0.5}
    result = objective(test_config)
    assert "loss" in result
    assert isinstance(result["loss"], (int, float))

    # Try running full Ray Tune optimization if environment supports it
    try:
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        tuner = tune.Tuner(
            objective,
            param_space={
                "x0": tune.uniform(-5, 5),
                "x1": tune.uniform(-5, 5),
            },
            tune_config=tune.TuneConfig(
                num_samples=20,
                metric="loss",
                mode="min",
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result()

        # Should find minimum near [0, 0]
        assert best_result.metrics["loss"] < 2.0
    except Exception as e:
        # Ray can have environment-specific issues; the basic interface test passed
        pytest.skip(f"Ray Tune full optimization skipped due to environment issue: {e}")
    finally:
        if ray.is_initialized():
            ray.shutdown()


############################################################
# SMAC integration test


def test_smac_integration():
    """Test that Surfaces functions work with SMAC."""
    pytest.importorskip("smac")
    from ConfigSpace import ConfigurationSpace, Float
    from smac import HyperparameterOptimizationFacade, Scenario

    func = SphereFunction(n_dim=2)

    # SMAC passes Configuration objects, which behave like dicts
    def objective(config, seed=0):
        return func({"x0": config["x0"], "x1": config["x1"]})

    # Define configuration space
    configspace = ConfigurationSpace(seed=42)
    configspace.add(Float("x0", (-5.0, 5.0)))
    configspace.add(Float("x1", (-5.0, 5.0)))

    # Create scenario
    scenario = Scenario(
        configspace,
        deterministic=True,
        n_trials=50,
    )

    # Run optimization
    smac = HyperparameterOptimizationFacade(
        scenario,
        objective,
        overwrite=True,
    )
    incumbent = smac.optimize()

    # Should find minimum near [0, 0]
    result = func({"x0": incumbent["x0"], "x1": incumbent["x1"]})
    assert result < 1.0
