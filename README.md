# Surfaces

[![PyPI version](https://img.shields.io/pypi/v/surfaces.svg)](https://pypi.org/project/surfaces/)
[![Python versions](https://img.shields.io/pypi/pyversions/surfaces.svg)](https://pypi.org/project/surfaces/)
[![License](https://img.shields.io/pypi/l/surfaces.svg)](https://github.com/SimonBlanke/Surfaces/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/surfaces/month)](https://pepy.tech/project/surfaces)

A collection of objective functions for optimization algorithm benchmarking.

## Overview

Surfaces provides standardized test functions for evaluating and comparing optimization algorithms. The library includes algebraic test functions (mathematical benchmark problems), machine learning test functions (hyperparameter optimization scenarios), and engineering design problems.

## Installation

```bash
pip install surfaces
```

For machine learning test functions (requires scikit-learn):

```bash
pip install surfaces[ml]
```

## Function Categories

### Algebraic Functions

Algebraic benchmark functions from optimization literature. These functions have known optima and well-characterized difficulty properties.

**1D Functions:**
- `GramacyAndLeeFunction`

**2D Functions:**
- `AckleyFunction`, `BealeFunction`, `BoothFunction`, `BukinFunctionN6`
- `CrossInTrayFunction`, `DropWaveFunction`, `EasomFunction`, `EggholderFunction`
- `GoldsteinPriceFunction`, `HimmelblausFunction`, `HoelderTableFunction`
- `LangermannFunction`, `LeviFunctionN13`, `MatyasFunction`, `McCormickFunction`
- `SchafferFunctionN2`, `SimionescuFunction`, `ThreeHumpCamelFunction`

**N-Dimensional Functions:**
- `GriewankFunction`, `RastriginFunction`, `RosenbrockFunction`
- `SphereFunction`, `StyblinskiTangFunction`

### Machine Learning Functions

Hyperparameter optimization problems based on scikit-learn models. Each function evaluates model performance via cross-validation on built-in datasets.

**Classification (5 models x 5 datasets):**

| Model | Hyperparameters |
|-------|-----------------|
| `KNeighborsClassifierFunction` | n_neighbors, algorithm |
| `DecisionTreeClassifierFunction` | max_depth, min_samples_split, min_samples_leaf |
| `RandomForestClassifierFunction` | n_estimators, max_depth, min_samples_split |
| `GradientBoostingClassifierFunction` | n_estimators, max_depth, learning_rate |
| `SVMClassifierFunction` | C, kernel, gamma |

**Regression (5 models x 5 datasets):**

| Model | Hyperparameters |
|-------|-----------------|
| `KNeighborsRegressorFunction` | n_neighbors, algorithm |
| `DecisionTreeRegressorFunction` | max_depth, min_samples_split, min_samples_leaf |
| `RandomForestRegressorFunction` | n_estimators, max_depth, min_samples_split |
| `GradientBoostingRegressorFunction` | n_estimators, max_depth |
| `SVMRegressorFunction` | C, kernel, gamma |

**Datasets:**
- Classification: digits, iris, wine, breast_cancer, covtype
- Regression: diabetes, california, friedman1, friedman2, linear

### Engineering Functions

Constrained engineering design optimization problems with penalty-based constraint handling.

- `CantileverBeamFunction` - Cantilever beam weight minimization
- `PressureVesselFunction` - Cylindrical pressure vessel cost minimization
- `TensionCompressionSpringFunction` - Spring weight minimization
- `ThreeBarTrussFunction` - Truss structure weight minimization
- `WeldedBeamFunction` - Welded beam fabrication cost minimization

## Usage

### Algebraic Function

```python
from surfaces.test_functions import SphereFunction

# Create function
sphere = SphereFunction(n_dim=3, objective="minimize")

# Get search space
print(sphere.search_space)
# {'x0': array([-5.12, ..., 5.12]), 'x1': array([...]), 'x2': array([...])}

# Evaluate
result = sphere({"x0": 0.5, "x1": -0.3, "x2": 0.1})
print(result)  # 0.35
```

### Machine Learning Function

```python
from surfaces.test_functions import KNeighborsClassifierFunction

# Create function with fixed dataset and cv
knn = KNeighborsClassifierFunction(dataset="iris", cv=5)

# Search space contains only hyperparameters
print(knn.search_space)
# {'n_neighbors': [3, 8, 13, ...], 'algorithm': ['auto', 'ball_tree', ...]}

# Evaluate
accuracy = knn({"n_neighbors": 5, "algorithm": "auto"})
print(f"Accuracy: {accuracy:.4f}")
```

### Engineering Function

```python
from surfaces.test_functions import WeldedBeamFunction

# Create function
beam = WeldedBeamFunction(objective="minimize")

# Evaluate with constraint penalties
cost = beam({"h": 0.2, "l": 6.0, "t": 8.0, "b": 0.3})
```

### Integration with Optimizers

```python
from surfaces.test_functions import AckleyFunction

# Works with any optimizer that accepts callable + search space
func = AckleyFunction(objective="minimize")

# Example with Hyperactive
from hyperactive import Hyperactive

hyper = Hyperactive()
hyper.add_search(func, func.search_space, n_iter=100)
hyper.run()
```

## Common Parameters

All test functions support:

| Parameter | Description |
|-----------|-------------|
| `objective` | "minimize" or "maximize" |
| `sleep` | Artificial delay per evaluation (seconds) |
| `memory` | Cache repeated evaluations |
| `collect_data` | Store evaluation history |
| `callbacks` | List of callback functions |

N-dimensional functions additionally support:
| Parameter | Description |
|-----------|-------------|
| `n_dim` | Number of dimensions |

ML functions additionally support:
| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset name (e.g., "iris", "diabetes") |
| `cv` | Cross-validation folds (2, 3, 5, or 10) |

## Visualizations

<table>
  <tr>
    <th>Function</th>
    <th>Heatmap</th>
    <th>Surface</th>
  </tr>
  <tr>
    <td><b>Sphere</b></td>
    <td><img src="./doc/images/mathematical/sphere_function_heatmap.jpg" width="90%"></td>
    <td><img src="./doc/images/mathematical/sphere_function_surface.jpg" width="100%"></td>
  </tr>
  <tr>
    <td><b>Rastrigin</b></td>
    <td><img src="./doc/images/mathematical/rastrigin_function_heatmap.jpg" width="90%"></td>
    <td><img src="./doc/images/mathematical/rastrigin_function_surface.jpg" width="100%"></td>
  </tr>
  <tr>
    <td><b>Ackley</b></td>
    <td><img src="./doc/images/mathematical/ackley_function_heatmap.jpg" width="90%"></td>
    <td><img src="./doc/images/mathematical/ackley_function_surface.jpg" width="100%"></td>
  </tr>
  <tr>
    <td><b>Rosenbrock</b></td>
    <td><img src="./doc/images/mathematical/rosenbrock_function_heatmap.jpg" width="90%"></td>
    <td><img src="./doc/images/mathematical/rosenbrock_function_surface.jpg" width="100%"></td>
  </tr>
</table>

## Roadmap

Planned additions:

- **BBOB Functions**: Black-Box Optimization Benchmarking suite (24 functions)
- **CEC Functions**: Competition on Evolutionary Computation benchmark suites
- **Surrogate Models**: Pre-trained ONNX models for fast ML function approximation
- **Additional ML Models**: Neural networks, XGBoost, LightGBM
- **Multi-objective Functions**: Test functions with multiple objectives

## License

MIT License
