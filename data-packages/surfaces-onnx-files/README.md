# surfaces-onnx-files

Pre-trained ONNX surrogate models for the [surfaces](https://github.com/SimonBlanke/Surfaces) library.

## Installation

This package is typically installed as a dependency of surfaces:

```bash
pip install surfaces[surrogates]
```

Or install directly:

```bash
pip install surfaces-onnx-files
```

## Contents

This package provides pre-trained ONNX models for fast surrogate evaluation of machine learning test functions:

| Model | Description |
|-------|-------------|
| k_neighbors_regressor | KNN regressor surrogate |
| k_neighbors_classifier | KNN classifier surrogate |
| gradient_boosting_regressor | Gradient boosting regressor surrogate |

## Usage

Once installed, the surfaces library automatically detects and uses this package:

```python
from surfaces import load_surrogate

surrogate = load_surrogate("k_neighbors_regressor")
if surrogate:
    result = surrogate.predict({"n_neighbors": 5, "leaf_size": 30, ...})
```

## License

MIT License - See [LICENSE](https://github.com/SimonBlanke/Surfaces/blob/main/LICENSE).
