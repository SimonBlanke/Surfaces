# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- README badges for PyPI, Python versions, license, and downloads
- CITATION.cff for academic citations
- py.typed marker for PEP 561 type checking support
- Type annotations with 84% coverage (682 of 813 functions):
  - Base test function classes
  - All algebraic functions (1D, 2D, ND)
  - BBOB benchmark functions
  - CEC benchmark functions (2013, 2014, 2017)
  - Machine learning functions (tabular, image, timeseries)
  - Engineering design functions
  - Noise module
  - Visualization utilities

## [0.5.1] - 2024-05-27

### Changed
- Moved `create_objective_function` into decorator for cleaner API

### Fixed
- Objective function name handling

## [0.5.0] - 2024-05-26

### Changed
- Complete restructure of ML test functions
- Centralized default arguments using `*args, **kwargs` schema
- Moved `evaluate_from_data` parameter to ML class only
- Renamed base class file for clarity

### Removed
- Removed `input_type` argument from API

## [0.4.0] - 2024-05-09

### Added
- Examples for common use cases
- `load` method to SurfacesDataCollector
- `max`/`min` parameter to search space configuration
- List of all available test functions

### Changed
- Complete API restructure with three init files for cleaner imports
- Moved to src-layout project structure
- Migrated from setup.py to pyproject.toml
- Search space creation with fixed size
- Moved data collection functions to separate module
- Class attributes moved to class level

### Fixed
- Positional and keyword args handling in ML/math test functions
- Rosenbrock and Griewank test function implementations
- KNN parameters

## [0.3.0] - 2024-03-16

### Added
- New `search_space` method to N-dimensional test functions
- Code coverage reporting in CI

### Changed
- Adapted all test functions to new API
- Renamed method to `search_space_from_blank`

## [0.2.2] - 2024-03-01

### Fixed
- Search data collector version requirement

## [0.2.1] - 2024-02-20

### Fixed
- Search data handling when None

## [0.2.0] - 2024-02-13

### Added
- Search space methods for test functions
- Specific search spaces per test function
- MANIFEST.in for package data
- Docker-based testing infrastructure
- Linter integration in CI

### Changed
- Separated 1D, 2D, ND test functions into directories
- Create `search_space` from `search_space_blank`

### Removed
- Python 3.7 support

## [0.1.1] - 2024-01-15

### Added
- Name-class attributes to ML functions
- Unique search space per function

### Changed
- Split public API into 1D, 2D, ND and machine learning test functions
- Evaluate loss instead of score for mathematical test functions
- Separated `search_data` and `collect_data` methods

### Fixed
- Search data collector version requirement
- Gramacy and Lee function search space

## [0.1.0] - 2024-01-01

### Added
- Initial release with algebraic test functions
- Machine learning test functions (classification and regression)
- Engineering design optimization problems
- Visualization capabilities (heatmap, surface plots)
- Memory caching for repeated evaluations
- Callback system for custom logging
- Data collection functionality

### Algebraic Functions
- 1D: GramacyAndLeeFunction
- 2D: Ackley, Beale, Booth, Bukin N6, CrossInTray, DropWave, Easom, Eggholder, GoldsteinPrice, Himmelblaus, HoelderTable, Langermann, Levi N13, Matyas, McCormick, Schaffer N2, Simionescu, ThreeHumpCamel
- ND: Griewank, Rastrigin, Rosenbrock, Sphere, StyblinskiTang

### Machine Learning Functions
- Classification: KNeighbors, DecisionTree, RandomForest, GradientBoosting, SVM
- Regression: KNeighbors, DecisionTree, RandomForest, GradientBoosting, SVM
- Datasets: digits, iris, wine, breast_cancer, covtype, diabetes, california, friedman1, friedman2, linear

### Engineering Functions
- CantileverBeam, PressureVessel, TensionCompressionSpring, ThreeBarTruss, WeldedBeam

[Unreleased]: https://github.com/SimonBlanke/Surfaces/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/SimonBlanke/Surfaces/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/SimonBlanke/Surfaces/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/SimonBlanke/Surfaces/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/SimonBlanke/Surfaces/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/SimonBlanke/Surfaces/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/SimonBlanke/Surfaces/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/SimonBlanke/Surfaces/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/SimonBlanke/Surfaces/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/SimonBlanke/Surfaces/releases/tag/v0.1.0
