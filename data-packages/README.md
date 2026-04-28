# Data Packages

This directory contains separate PyPI packages for large data files that are optional dependencies of the main `surfaces` package. This architecture keeps the core package lean (~230KB) while allowing users to install additional data as needed.

## Overview

```
data-packages/
└── surfaces-cec-data/      # CEC benchmark rotation matrices & shift vectors (~6.3MB)
```

ONNX surrogate models were previously shipped here as `surfaces-onnx-files`. They have moved to the standalone `surfaces-surrogates` package (see https://github.com/SimonBlanke/surfaces-surrogates).

## surfaces-cec-data

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

## File Lookup Order

The CEC data package uses a two-tier lookup system (`_data_utils.py`):

```
1. Local data package (development)
   data-packages/surfaces-cec-data/src/surfaces_cec_data/{dataset}/{file}

2. Installed surfaces-cec-data package
   via importlib.resources
```

This lookup order ensures developers can work without installing the data package, while end users get data from the installed package.

## Development Workflow

During development, the lookup finds files in the local `data-packages/` directory automatically. No installation required.

```python
from surfaces.test_functions.cec.cec2014 import RotatedHighConditionedElliptic

func = RotatedHighConditionedElliptic(n_dim=10)
result = func([0] * 10)
```

## Building and Publishing

```bash
cd data-packages/surfaces-cec-data
python -m build --wheel
# Output: dist/surfaces_cec_data-0.1.0-py3-none-any.whl (~6.3MB)

twine upload dist/*
```

## Package Sizes

| Package | Size | Contents |
|---------|------|----------|
| surfaces | ~230KB | Core library code |
| surfaces-cec-data | ~6.3MB | CEC benchmark data |
| surfaces-surrogates | ~60KB | ONNX surrogate models + training tools (separate repo) |
