# Data Packages

This directory contains separate PyPI packages for large data files that are optional dependencies of the main `surfaces` package. This architecture keeps the core package lean (~230KB) while allowing users to install additional data as needed.

## Overview

```
data-packages/
├── surfaces-cec-data/      # CEC benchmark rotation matrices & shift vectors (~6.3MB)
└── surfaces-onnx-files/    # Pre-trained ONNX surrogate models (~57KB)
```

## Packages

### surfaces-cec-data

Contains `.npz` files with rotation matrices, shift vectors, and shuffle indices for CEC (Congress on Evolutionary Computation) benchmark functions.

| Suite | Dimensions | Functions |
|-------|------------|-----------|
| CEC 2013 | 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 | F1-F28 |
| CEC 2014 | 10, 20, 30, 50, 100 | F1-F30 |
| CEC 2017 | 2, 10, 20, 30, 50, 100 | F1-F30 |

**Installation:**
```bash
pip install surfaces[cec]
# or directly:
pip install surfaces-cec-data
```

### surfaces-onnx-files

Contains pre-trained ONNX surrogate models for fast approximation of ML test functions.

| Model | Description |
|-------|-------------|
| k_neighbors_regressor | KNN regressor surrogate |
| k_neighbors_classifier | KNN classifier surrogate |
| gradient_boosting_regressor | Gradient boosting regressor surrogate |

**Installation:**
```bash
pip install surfaces[surrogates]
# or directly:
pip install surfaces-onnx-files onnxruntime
```

## File Lookup Order

Both packages use a three-tier lookup system that checks locations in order:

### CEC Data (`_data_utils.py`)

```
1. Local data package (development)
   → data-packages/surfaces-cec-data/src/surfaces_cec_data/{dataset}/{file}

2. Installed surfaces-cec-data package
   → via importlib.resources
```

### ONNX Files (`_onnx_utils.py`)

```
1. Local models directory (newly trained models)
   → src/surfaces/_surrogates/models/{file}

2. Local data package (development)
   → data-packages/surfaces-onnx-files/src/surfaces_onnx_files/{file}

3. Installed surfaces-onnx-files package
   → via importlib.resources
```

This lookup order ensures:
- Developers can work without installing data packages
- Newly trained models take precedence over shipped ones
- End users get data from installed packages

## Architecture Diagram

```
surfaces (main package)
│
├── test_functions/cec/
│   ├── _data_utils.py          ──────┐
│   └── _base_cec.py                  │
│       └── _load_data()              │
│           └── get_data_file() ──────┼──→ surfaces-cec-data
│                                     │    (PyPI package)
│
└── _surrogates/
    ├── _onnx_utils.py          ──────┐
    ├── _surrogate_loader.py          │
    │   └── get_surrogate_path() ─────┼──→ surfaces-onnx-files
    └── _dashboard/                   │    (PyPI package)
        └── sync.py                   │
            └── get_metadata_path() ──┘
```

## Development Workflow

### Working with CEC data

During development, the lookup finds files in the local `data-packages/` directory automatically. No installation required.

```python
from surfaces.test_functions.cec.cec2014 import RotatedHighConditionedElliptic

# Works without installing surfaces-cec-data
func = RotatedHighConditionedElliptic(n_dim=10)
result = func([0] * 10)
```

### Working with ONNX surrogates

Similarly, ONNX files are found locally during development:

```python
from surfaces._surrogates import load_surrogate

# Works without installing surfaces-onnx-files
surrogate = load_surrogate("k_neighbors_regressor")
```

### Training new surrogates

New models are saved to `src/surfaces/_surrogates/models/` and take precedence:

```python
from surfaces._surrogates import train_ml_surrogate

# Trains and exports to local models/ directory
train_ml_surrogate("gradient_boosting_classifier")

# New model is immediately available (no package rebuild needed)
surrogate = load_surrogate("gradient_boosting_classifier")
```

After training, copy new models to the data package for distribution:
```bash
cp src/surfaces/_surrogates/models/new_model.* \
   data-packages/surfaces-onnx-files/src/surfaces_onnx_files/
```

## Building Packages

### Build surfaces-cec-data

```bash
cd data-packages/surfaces-cec-data
python -m build --wheel
# Output: dist/surfaces_cec_data-0.1.0-py3-none-any.whl (~6.3MB)
```

### Build surfaces-onnx-files

```bash
cd data-packages/surfaces-onnx-files
python -m build --wheel
# Output: dist/surfaces_onnx_files-0.1.0-py3-none-any.whl (~57KB)
```

### Publish to PyPI

```bash
cd data-packages/surfaces-cec-data
twine upload dist/*

cd data-packages/surfaces-onnx-files
twine upload dist/*
```

## Error Handling

When data is not available (neither locally nor via installed package), helpful error messages guide the user:

```python
# CEC functions without data:
ImportError: CEC benchmark data files are not available.
Install the data package with: pip install surfaces-cec-data
Or install surfaces with CEC support: pip install surfaces[cec]

# Surrogates without ONNX files:
# Returns None (graceful degradation)
surrogate = load_surrogate("missing_model")  # Returns None
```

## Package Sizes

| Package | Size | Contents |
|---------|------|----------|
| surfaces | ~230KB | Core library code |
| surfaces-cec-data | ~6.3MB | CEC benchmark data |
| surfaces-onnx-files | ~57KB | ONNX surrogate models |

The main `surfaces` package went from ~8.8MB to ~230KB by extracting data files.
