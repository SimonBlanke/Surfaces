# Surrogate Models for Surfaces

This module provides neural network surrogate models for fast approximation of expensive ML test functions.

## Overview

Machine learning test functions (like `KNeighborsClassifierFunction`) require actual model training and cross-validation, which can take hundreds of milliseconds per evaluation. Surrogate models approximate these functions using pre-trained neural networks, enabling:

- **~10,000 evaluations/second** instead of ~1-10 evaluations/second
- **Speedups of 100x-6000x** depending on the function
- **R² > 0.94** approximation accuracy

## Quick Start

### Using Pre-trained Surrogates

```python
from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

# Enable surrogate mode with a single parameter
func = KNeighborsClassifierFunction(use_surrogate=True)

# Use exactly like the real function
result = func({
    'n_neighbors': 5,
    'algorithm': 'auto',
    'cv': 5,
    'dataset': digits_data,
})
```

If no surrogate exists for a function, it falls back to real evaluation with a warning.

### Available Surrogates

| Function | Model Size | R² Score | Speedup |
|----------|------------|----------|---------|
| `KNeighborsClassifierFunction` | 19 KB | 0.97 | ~150x |
| `GradientBoostingRegressorFunction` | 19 KB | 0.99 | ~2,400x |

## Architecture

```
_surrogates/
├── __init__.py                 # Public exports
├── _surrogate_loader.py        # Load and run ONNX models
├── _surrogate_trainer.py       # Train new surrogates (maintainer use)
├── _surrogate_validator.py     # Validate surrogate accuracy
├── README.md                   # This file
└── models/                     # Pre-trained ONNX files
    ├── k_neighbors_classifier.onnx
    ├── k_neighbors_classifier.onnx.meta.json
    ├── gradient_boosting_regressor.onnx
    └── gradient_boosting_regressor.onnx.meta.json
```

## How It Works

### Training Pipeline

1. **Sample Collection**: Grid-sample the hyperparameter space, evaluate with real function
2. **Preprocessing**: Encode categorical parameters, normalize numeric parameters
3. **Training**: Fit sklearn MLPRegressor on collected samples
4. **Export**: Convert to ONNX format with metadata

### Inference Pipeline

1. **Load**: ONNX model loaded via onnxruntime (lazy, on first use)
2. **Encode**: Convert parameter dict to numeric array using metadata
3. **Predict**: Run ONNX inference (~0.1ms)
4. **Return**: Surrogate prediction used in place of real evaluation

### Parameter Handling

The surrogate handles different parameter types:

| Type | Example | Handling |
|------|---------|----------|
| Numeric | `n_neighbors=5` | Pass through as float |
| Categorical | `algorithm='auto'` | Encode as integer via metadata |
| Callable | `dataset=digits_data` | Encode by function name |

## Training New Surrogates (Maintainer Guide)

### Basic Training

```python
from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
from surfaces._surrogates import SurrogateTrainer

# Create the expensive function
func = KNeighborsClassifierFunction()

# Initialize trainer
trainer = SurrogateTrainer(func)

# Collect training samples (grid-based)
trainer.collect_samples_grid(max_samples=1000, verbose=True)

# Train MLP
trainer.train(hidden_layer_sizes=(64, 64), max_iter=1000)

# Export to ONNX
trainer.export('k_neighbors_classifier.onnx')
```

### Training Parameters

```python
# Sample collection
trainer.collect_samples_grid(
    max_samples=1000,    # Limit samples if grid is large
    verbose=True,        # Print progress
)

# Training
trainer.train(
    hidden_layer_sizes=(64, 64),  # MLP architecture
    max_iter=1000,                # Training iterations
    verbose=True,                 # Print training progress
)
```

### Convenience Function

```python
from surfaces._surrogates import train_surrogate_for_function

path = train_surrogate_for_function(
    function=KNeighborsClassifierFunction(),
    output_name='k_neighbors_classifier',
    max_samples=1000,
    hidden_layers=(64, 64),
)
```

## Validating Surrogates

### Basic Validation

```python
from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction
from surfaces._surrogates import SurrogateValidator

func = KNeighborsClassifierFunction()
validator = SurrogateValidator(func)

# Random sampling validation
results = validator.validate_random(n_samples=100, seed=42)

# Grid sampling validation
results = validator.validate_grid(max_samples=100)

# Run both
results = validator.summary(n_random=100, n_grid=100)
```

### Validation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| R² | Coefficient of determination | > 0.90 |
| Correlation | Pearson correlation | > 0.95 |
| MAE | Mean absolute error | < 0.05 |
| RMSE | Root mean squared error | < 0.05 |
| Max Error | Worst-case error | < 0.15 |

### Interpreting Results

```
==================================================
Surrogate Validation Results (random sampling)
==================================================
Samples evaluated: 100

Accuracy Metrics
  R² Score:     0.9657    # Excellent: explains 97% of variance
  Correlation:  0.9834    # Excellent: rankings preserved
  MAE:          0.0157    # Good: avg error ~1.5%
  RMSE:         0.0241    # Good: typical error ~2.4%
  Max Error:    0.0814    # Acceptable: worst case ~8%

Timing
  Real function:    42.36 ms/eval
  Surrogate:         0.29 ms/eval
  Speedup:            149x
```

## Dependencies

### For Using Surrogates (Inference)

```bash
pip install surfaces[surrogate]
```

Installs: `onnxruntime>=1.16.0`

### For Training Surrogates

```bash
pip install surfaces[surrogate-train]
```

Installs: `onnxruntime>=1.16.0`, `skl2onnx>=1.16.0`

## Metadata Format

Each ONNX model has an accompanying `.meta.json` file:

```json
{
  "function_name": "k_neighbors_classifier",
  "param_names": ["algorithm", "cv", "dataset", "n_neighbors"],
  "param_encodings": {
    "algorithm": {"auto": 0, "ball_tree": 1, "kd_tree": 2, "brute": 3},
    "dataset": {"digits_data": 0, "wine_data": 1, "iris_data": 2}
  },
  "n_samples": 866,
  "y_range": [0.399, 0.980],
  "training_time": 1.09,
  "collection_time": 40.41
}
```

## Validity Model

Some hyperparameter combinations are invalid (e.g., `n_neighbors > dataset_size` in KNN). The real function returns `NaN` for these cases.

The surrogate system handles this by training a **validity classifier** alongside the regression model:

1. During training, both valid and invalid samples are collected
2. A binary classifier learns to predict validity
3. During inference, validity is checked first
4. Invalid combinations return `NaN`, just like the real function

```python
# Surrogate correctly returns NaN for invalid combinations
func = KNeighborsClassifierFunction(use_surrogate=True)

# Valid: returns score
result = func({'n_neighbors': 5, 'cv': 5, 'dataset': digits_data, ...})
# 0.9560

# Invalid: returns NaN (n_neighbors too large for dataset)
result = func({'n_neighbors': 140, 'cv': 5, 'dataset': iris_data, ...})
# nan
```

Files for a function with validity model:
```
models/
├── k_neighbors_classifier.onnx              # Regression model
├── k_neighbors_classifier.validity.onnx     # Validity classifier
└── k_neighbors_classifier.onnx.meta.json    # Metadata (has_validity_model: true)
```

## Limitations

1. **Interpolation only**: Surrogates work best within the training search space
2. **No uncertainty**: Point estimates only (no confidence intervals)
3. **Fixed search space**: Changing the search space requires retraining
4. **Approximation errors**: ~1-5% typical error, up to ~10% worst case

## Adding a New Surrogate

1. Train and validate:
   ```python
   func = NewMLFunction()
   trainer = SurrogateTrainer(func)
   trainer.collect_samples_grid(max_samples=1000)
   trainer.train()
   trainer.export('new_ml_function.onnx')

   validator = SurrogateValidator(func)
   validator.summary()  # Verify R² > 0.90
   ```

2. Ensure the function has `_name_` attribute matching the ONNX filename

3. Place ONNX and metadata files in `models/` directory

4. Test the integration:
   ```python
   func = NewMLFunction(use_surrogate=True)
   assert func.use_surrogate == True  # Should not fall back
   ```
