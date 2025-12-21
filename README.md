<H1 align="center">
    Surfaces
</H1>

<br>

<H2 align="center">
    A collection of single-objective black-box functions for optimization benchmarking
</H2>

<br>

## Introduction

**Surfaces** is a Python library designed for researchers and practitioners in optimization and machine learning. It provides a comprehensive collection of benchmark functions to evaluate and compare optimization algorithms under standardized conditions.

### Key Features

- **Mathematical Test Functions**: 24 test functions covering 1D, 2D, and N-dimensional optimization landscapes
- **Machine Learning Test Functions**: Real-world ML benchmarks using scikit-learn models with hyperparameter optimization
- **Standardized Interface**: Consistent API across all functions for easy algorithm evaluation
- **Flexible Configuration**: Support for different metrics (loss/score), dimensionalities, and evaluation modes
- **Optimization Research**: Ideal for testing metaheuristics, gradient-free methods, and hyperparameter optimization algorithms

### Use Cases

- **Algorithm Development**: Test new optimization algorithms against established benchmarks
- **Performance Comparison**: Compare different optimizers on standardized problem sets
- **Research Publications**: Use well-known test functions with consistent implementations
- **Educational Purposes**: Learn optimization concepts with visual and mathematical examples
- **Hyperparameter Tuning**: Benchmark autoML and hyperparameter optimization methods


<br>

## Visualizations

### Mathematical Functions

<table style="width:100%">
  <tr>
    <th> <b>Objective Function</b> </th>
    <th> <b>Heatmap</b> </th>
    <th> <b>Surface Plot</b> </th>
  </tr>
  <tr>
    <th> <ins>Sphere function</ins> <br><br>  </th>
    <td> <img src="./doc/images/mathematical/sphere_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/sphere_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Rastrigin function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/rastrigin_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/rastrigin_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Ackley function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/ackley_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/ackley_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Rosenbrock function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/rosenbrock_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/rosenbrock_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Beale function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/beale_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/beale_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Himmelblaus function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/himmelblaus_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/himmelblaus_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Hölder Table function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/hölder_table_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/hölder_table_function_surface.jpg" width="100%"> </td>
  </tr>
  <tr>
    <th> <ins>Cross-In-Tray function</ins> <br><br> </th>
    <td> <img src="./doc/images/mathematical/cross_in_tray_function_heatmap.jpg" width="90%"> </td>
    <td> <img src="./doc/images/mathematical/cross_in_tray_function_surface.jpg" width="100%"> </td>
  </tr>
</table>


<br>

## Installation

The most recent version of Surfaces is available on PyPi:

```console
pip install surfaces
```

## Public API

### Test Functions

Surfaces provides two main categories of test functions for optimization benchmarking:

#### Mathematical Functions

Import from `surfaces.test_functions.mathematical`:

**1D Functions:**
- `GramacyAndLeeFunction`

**2D Functions:**
- `AckleyFunction`, `BealeFunction`, `BoothFunction`, `BukinFunctionN6`
- `CrossInTrayFunction`, `DropWaveFunction`, `EasomFunction`, `EggholderFunction`
- `GoldsteinPriceFunction`, `HimmelblausFunction`, `HölderTableFunction`
- `LangermannFunction`, `LeviFunctionN13`, `MatyasFunction`, `McCormickFunction`
- `SchafferFunctionN2`, `SimionescuFunction`, `ThreeHumpCamelFunction`

**N-Dimensional Functions:**
- `GriewankFunction`, `RastriginFunction`, `RosenbrockFunction`
- `SphereFunction`, `StyblinskiTangFunction`

#### Machine Learning Functions

Import from `surfaces.test_functions.machine_learning`:

- `KNeighborsClassifierFunction` - K-nearest neighbors classification
- `KNeighborsRegressorFunction` - K-nearest neighbors regression
- `GradientBoostingRegressorFunction` - Gradient boosting regression

### Common Interface

All test functions provide:

- `objective_function(parameters)` - Evaluate the function
- `search_space()` - Get parameter bounds/ranges
- Constructor parameters:
  - `metric="loss"` or `"score"` - Optimization direction
  - `sleep=0` - Add artificial delays for benchmarking
  - `n_dim=N` (for N-dimensional functions) - Set dimensionality

## Usage Examples

### Basic Mathematical Function

```python
from surfaces.test_functions.mathematical import SphereFunction, AckleyFunction

# Create functions
sphere_function = SphereFunction(n_dim=2, metric="score")
ackley_function = AckleyFunction(metric="loss")

# Get search space
search_space = sphere_function.search_space()
print(search_space)  # {'x0': array([...]), 'x1': array([...])}

# Evaluate function
params = {'x0': 0.5, 'x1': -0.3}
result = sphere_function.objective_function(params)
print(f"Sphere function result: {result}")
```

### Machine Learning Function

```python
from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

# Create ML function
knn_func = KNeighborsClassifierFunction(metric="accuracy")

# Get search space
search_space = knn_func.search_space()
print(search_space.keys())  # dict_keys(['n_neighbors', 'algorithm', 'cv', 'dataset'])

# Evaluate function
params = {
    'n_neighbors': 5,
    'algorithm': 'auto',
    'cv': 3,
    'dataset': knn_func.search_space()['dataset'][0]
}
accuracy = knn_func.objective_function(params)
print(f"KNN accuracy: {accuracy}")
```

### N-Dimensional Functions

```python
from surfaces.test_functions.mathematical import SphereFunction

# Create functions with different dimensionalities
sphere_1d = SphereFunction(n_dim=1)
sphere_3d = SphereFunction(n_dim=3)
sphere_10d = SphereFunction(n_dim=10)

# Each has appropriate search space
print(sphere_1d.search_space().keys())   # dict_keys(['x0'])
print(sphere_3d.search_space().keys())   # dict_keys(['x0', 'x1', 'x2'])
print(sphere_10d.search_space().keys())  # dict_keys(['x0', 'x1', ..., 'x9'])
```
