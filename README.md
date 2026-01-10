<p align="center">
  <a href="https://github.com/SimonBlanke/Surfaces">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="./docs/source/_static/surfaces_logo_dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="./docs/source/_static/surfaces_logo.svg">
      <img src="./docs/source/_static/surfaces_logo.svg" width="400" alt="Surfaces Logo">
    </picture>
  </a>
</p>

---

<h3 align="center">
Test functions for benchmarking optimization algorithms in Python.
</h3>

<p align="center">
  <a href="https://github.com/SimonBlanke/Surfaces/actions"><img src="https://img.shields.io/github/actions/workflow/status/SimonBlanke/Surfaces/tests.yml?style=flat-square&label=tests" alt="Tests"></a>
  <a href="https://codecov.io/gh/SimonBlanke/Surfaces"><img src="https://img.shields.io/codecov/c/github/SimonBlanke/Surfaces?style=flat-square" alt="Coverage"></a>
</p>

<table align="center">
  <tr>
    <th align="right">Documentation</th>
    <th align="center">▸</th>
    <th align="left"><a href="https://github.com/SimonBlanke/Surfaces">Homepage</a> · <a href="https://github.com/SimonBlanke/Surfaces#usage">User Guide</a> · <a href="https://github.com/SimonBlanke/Surfaces#function-categories">API Reference</a> · <a href="https://github.com/SimonBlanke/Surfaces#examples">Tutorials</a></th>
  </tr>
  <tr>
    <td align="right"><b>On this page</b></td>
    <td align="center">▸</td>
    <td align="left"><a href="#key-features">Features</a> · <a href="#examples">Examples</a> · <a href="#core-concepts">Concepts</a> · <a href="#citation">Citation</a></td>
  </tr>
</table>

<br>

---

**Surfaces** is a Python library providing standardized test functions for evaluating and comparing optimization algorithms. It includes algebraic benchmark functions from optimization literature, machine learning hyperparameter optimization problems, and constrained engineering design problems.

Designed for researchers benchmarking new optimization algorithms, practitioners comparing optimizer performance, and educators teaching optimization concepts. All functions provide a consistent interface: callable with dictionary parameters and automatic search space generation.

<p>
  <a href="https://linkedin.com/in/simon-blanke"><img src="https://img.shields.io/badge/LinkedIn-Follow-0A66C2?style=flat-square&logo=linkedin" alt="LinkedIn"></a>
  <a href="https://github.com/sponsors/SimonBlanke"><img src="https://img.shields.io/badge/Sponsor-EA4AAA?style=flat-square&logo=githubsponsors&logoColor=white" alt="Sponsor"></a>
</p>

---

## Installation

```bash
pip install surfaces
```

<p>
  <a href="https://pypi.org/project/surfaces/"><img src="https://img.shields.io/pypi/v/surfaces?style=flat-square&color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/surfaces/"><img src="https://img.shields.io/pypi/pyversions/surfaces?style=flat-square" alt="Python"></a>
</p>

<details>
<summary>Optional dependencies</summary>

```bash
pip install surfaces[ml]          # Machine learning test functions (scikit-learn)
pip install surfaces[viz]         # Visualization tools (matplotlib, plotly)
pip install surfaces[cec]         # CEC competition benchmarks
pip install surfaces[timeseries]  # Time series functions (sktime)
pip install surfaces[full]        # All optional features
```

</details>

---

## Key Features

| [**30+ Algebraic Functions**](#algebraic-functions)<br><sub>Classic benchmarks from optimization literature: Sphere, Rastrigin, Ackley, Rosenbrock, and more.</sub> | [**ML Hyperparameter Surfaces**](#machine-learning-functions)<br><sub>Real hyperparameter optimization landscapes using scikit-learn models with cross-validation.</sub> | [**Engineering Problems**](#engineering-functions)<br><sub>Constrained optimization from engineering literature: welded beams, pressure vessels, spring design.</sub> |
| :--- | :--- | :--- |
| [**Function Modifiers**](#using-modifiers)<br><sub>Add noise, delays, or transformations to any function. Simulate real-world conditions.</sub> | [**BBOB and CEC Suites**](#benchmark-suites)<br><sub>Industry-standard benchmark suites used in optimization competitions.</sub> | [**Optimizer Integration**](#integration-with-optimizers)<br><sub>Works with any optimizer that accepts a callable and search space.</sub> |

---

## Quick Start

```python
from surfaces.test_functions.algebraic import SphereFunction

# Create a 3-dimensional Sphere function
sphere = SphereFunction(n_dim=3)

# Get the search space (NumPy arrays for each dimension)
print(sphere.search_space)
# {'x0': array([-5.12, ..., 5.12]), 'x1': array([...]), 'x2': array([...])}

# Evaluate at a point
result = sphere({"x0": 0.5, "x1": -0.3, "x2": 0.1})
print(f"Value: {result}")  # Value: -0.35 (negated for maximization)

# Access the global optimum
print(f"Optimum: {sphere.global_optimum}")  # Optimum at origin
```

**Output:**
```
{'x0': array([-5.12, ..., 5.12]), 'x1': array([...]), 'x2': array([...])}
Value: -0.35
Optimum: {'x0': 0.0, 'x1': 0.0, 'x2': 0.0}
```

---

## Core Concepts

```
                     SURFACES ARCHITECTURE

    ┌──────────────────┐    ┌──────────────────┐
    │   Test Function  │───>│   Search Space   │
    │   (Algebraic/    │    │   (NumPy arrays  │
    │    ML/Engineer)  │    │    per param)    │
    └──────────────────┘    └──────────────────┘
            │                       │
            v                       v
    ┌─────────────────────────────────────────┐
    │             Modifiers (optional)         │
    │    (Noise, Delay, Transformations)       │
    └─────────────────────────────────────────┘
            │
            v
    ┌─────────────────────────────────────────┐
    │              Optimizer                   │
    │    (Hyperactive, GFO, or any other)      │
    └─────────────────────────────────────────┘
```

**Test Function**: Callable object that evaluates parameter combinations. Returns a score (maximization by default).

**Search Space**: Dictionary mapping parameter names to valid values as NumPy arrays. Generated automatically from function bounds.

**Modifiers**: Optional pipeline of transformations (noise, delay) applied to function evaluations.

**Optimizer**: Any algorithm that can call the function with parameters from the search space.

---

## Examples

<details open>
<summary><b>Algebraic Functions</b></summary>

```python
from surfaces.test_functions.algebraic import (
    SphereFunction,
    RastriginFunction,
    AckleyFunction,
)

# Simple unimodal function
sphere = SphereFunction(n_dim=5)
print(sphere({"x0": 0, "x1": 0, "x2": 0, "x3": 0, "x4": 0}))  # Optimum: 0

# Highly multimodal function with many local optima
rastrigin = RastriginFunction(n_dim=3)
print(f"Search space bounds: {rastrigin.bounds}")

# Challenging function with a narrow global basin
ackley = AckleyFunction()  # 2D by default
print(f"Global optimum at: {ackley.global_optimum}")
```

</details>

<br>

<details>
<summary><b>Machine Learning Functions</b></summary>

```python
from surfaces.test_functions.machine_learning.tabular import (
    KNeighborsClassifierFunction,
    RandomForestClassifierFunction,
)

# KNN hyperparameter optimization on iris dataset
knn = KNeighborsClassifierFunction(dataset="iris", cv=5)
print(knn.search_space)
# {'n_neighbors': [3, 5, 7, ...], 'algorithm': ['auto', 'ball_tree', ...]}

accuracy = knn({"n_neighbors": 5, "algorithm": "auto"})
print(f"CV Accuracy: {accuracy:.4f}")

# Random Forest on wine dataset
rf = RandomForestClassifierFunction(dataset="wine", cv=3)
score = rf({"n_estimators": 100, "max_depth": 5, "min_samples_split": 2})
```

</details>

<br>

<details>
<summary><b>Engineering Functions</b></summary>

```python
from surfaces.test_functions.algebraic.constrained import (
    WeldedBeamFunction,
    PressureVesselFunction,
)

# Welded beam design optimization
beam = WeldedBeamFunction()
cost = beam({"h": 0.2, "l": 6.0, "t": 8.0, "b": 0.3})
print(f"Fabrication cost: {cost}")

# Pressure vessel design
vessel = PressureVesselFunction()
print(f"Design variables: {list(vessel.search_space.keys())}")
```

</details>

<br>

<details>
<summary><b>Using Modifiers</b></summary>

```python
from surfaces.test_functions.algebraic import SphereFunction
from surfaces.modifiers import DelayModifier, GaussianNoise

# Add realistic conditions to any function
sphere = SphereFunction(
    n_dim=2,
    modifiers=[
        DelayModifier(delay=0.01),         # Simulate expensive evaluation
        GaussianNoise(sigma=0.1, seed=42)  # Add measurement noise
    ]
)

# Evaluate with modifiers applied
noisy_result = sphere({"x0": 1.0, "x1": 2.0})

# Get the true value without modifiers
true_value = sphere.true_value({"x0": 1.0, "x1": 2.0})
print(f"Noisy: {noisy_result:.4f}, True: {true_value:.4f}")
```

</details>

<br>

<details>
<summary><b>Benchmark Suites</b></summary>

```python
from surfaces.test_functions.benchmark.bbob import (
    SphereFunction as BBOBSphere,
    RosenbrockFunction as BBOBRosenbrock,
)

# BBOB (Black-Box Optimization Benchmarking) functions
# Used in COCO platform for algorithm comparison
bbob_sphere = BBOBSphere(n_dim=10)
bbob_rosenbrock = BBOBRosenbrock(n_dim=10)

print(f"BBOB Sphere optimum: {bbob_sphere.optimal_value}")
```

</details>

<br>

<details>
<summary><b>Multi-Objective Functions</b></summary>

```python
from surfaces.test_functions.algebraic.multi_objective import (
    ZDT1,
    Kursawe,
)

# ZDT1: Two-objective problem with convex Pareto front
zdt1 = ZDT1(n_dim=30)
objectives = zdt1({"x0": 0.5, "x1": 0.0, "x2": 0.0})  # Returns tuple of objectives

# Kursawe: Non-convex Pareto front
kursawe = Kursawe()
f1, f2 = kursawe({"x0": 0.0, "x1": 0.0, "x2": 0.0})
```

</details>

<br>

<details>
<summary><b>Integration with Optimizers</b></summary>

```python
from surfaces.test_functions.algebraic import AckleyFunction

# Surfaces works with any optimizer accepting callable + search space
func = AckleyFunction()

# Example with Hyperactive
from hyperactive import Hyperactive

hyper = Hyperactive()
hyper.add_search(func, func.search_space, n_iter=100)
hyper.run()

print(f"Best score: {hyper.best_score(func)}")
print(f"Best params: {hyper.best_para(func)}")

# Example with Gradient-Free-Optimizers
from gradient_free_optimizers import BayesianOptimizer

opt = BayesianOptimizer(func.search_space)
opt.search(func, n_iter=50)
```

</details>

---

## Ecosystem

This library is part of a suite of optimization tools. For updates, [follow on GitHub](https://github.com/SimonBlanke).

| Package | Description |
|---------|-------------|
| [Hyperactive](https://github.com/SimonBlanke/Hyperactive) | Hyperparameter optimization framework with experiment abstraction and ML integrations |
| [Gradient-Free-Optimizers](https://github.com/SimonBlanke/Gradient-Free-Optimizers) | Core optimization algorithms for black-box function optimization |
| [Surfaces](https://github.com/SimonBlanke/Surfaces) | Test functions and benchmark surfaces for optimization algorithm evaluation |

---

## Documentation

| Resource | Description |
|----------|-------------|
| [User Guide](https://github.com/SimonBlanke/Surfaces#usage) | Installation and basic usage examples |
| [API Reference](https://github.com/SimonBlanke/Surfaces#function-categories) | Complete list of available test functions |
| [Examples](https://github.com/SimonBlanke/Surfaces#examples) | Code examples for common use cases |
| [GitHub Issues](https://github.com/SimonBlanke/Surfaces/issues) | Bug reports and feature requests |

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

- **Bug reports**: [GitHub Issues](https://github.com/SimonBlanke/Surfaces/issues)
- **Feature requests**: [GitHub Discussions](https://github.com/SimonBlanke/Surfaces/discussions)
- **Questions**: [GitHub Issues](https://github.com/SimonBlanke/Surfaces/issues)

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{surfaces2024,
  author = {Simon Blanke},
  title = {Surfaces: Test functions for optimization algorithm benchmarking},
  year = {2024},
  url = {https://github.com/SimonBlanke/Surfaces},
  version = {0.6.1}
}
```

---

## License

[MIT License](./LICENSE) - Free for commercial and academic use.
