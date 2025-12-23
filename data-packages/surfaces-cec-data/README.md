# surfaces-cec-data

CEC benchmark data files for the [surfaces](https://github.com/SimonBlanke/Surfaces) library.

## Installation

This package is typically installed as a dependency of surfaces:

```bash
pip install surfaces[cec]
```

Or install directly:

```bash
pip install surfaces-cec-data
```

## Contents

This package provides rotation matrices, shift vectors, and shuffle indices
for CEC (Congress on Evolutionary Computation) benchmark functions:

| Suite | Dimensions | Functions |
|-------|------------|-----------|
| CEC 2013 | 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100 | F1-F28 |
| CEC 2014 | 10, 20, 30, 50, 100 | F1-F30 |
| CEC 2017 | 2, 10, 20, 30, 50, 100 | F1-F30 |

## Usage

Once installed, the surfaces library automatically detects and uses this package:

```python
from surfaces.test_functions.cec.cec2017 import ShiftedRotatedBentCigar

func = ShiftedRotatedBentCigar(n_dim=10)
result = func([0.0] * 10)
```

## License

MIT License - See [LICENSE](https://github.com/SimonBlanke/Surfaces/blob/main/LICENSE).
