#!/usr/bin/env python
"""Generate CEC 2008 benchmark data files.

CEC 2008 is a large-scale optimization benchmark with 1000 dimensions.
This script generates:
- Shift vectors for all 7 functions (F1-F7)

The shift vectors are randomly generated within the bounds, ensuring the
global optimum is within the search space.

Usage:
    python scripts/create_cec2008_data.py
"""

from pathlib import Path

import numpy as np

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data-packages" / "surfaces-cec-data" / "src" / "surfaces_cec_data" / "cec2008"


def generate_shift_vector(dim: int, bounds: tuple, seed: int) -> np.ndarray:
    """Generate a random shift vector within bounds.

    The optimal point (shift vector) is placed within the inner 80% of the
    search space to avoid boundary effects.
    """
    np.random.seed(seed)
    lb, ub = bounds
    # Place optimum in inner 80% of search space
    margin = 0.1 * (ub - lb)
    return np.random.uniform(lb + margin, ub - margin, dim)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dim = 1000

    # CEC 2008 function specifications
    # Format: func_id, bounds, description
    functions = [
        (1, (-100, 100), "Shifted Sphere"),
        (2, (-100, 100), "Shifted Schwefel 2.21"),
        (3, (-100, 100), "Shifted Rosenbrock"),
        (4, (-5, 5), "Shifted Rastrigin"),
        (5, (-600, 600), "Shifted Griewank"),
        (6, (-32, 32), "Shifted Ackley"),
        (7, (-1, 1), "Fast Fractal Double Dip"),
    ]

    data = {}

    for func_id, bounds, name in functions:
        print(f"Generating shift vector for F{func_id}: {name}")
        # Use func_id as seed for reproducibility
        shift = generate_shift_vector(dim, bounds, seed=2008 * 100 + func_id)
        data[f"shift_{func_id}"] = shift.astype(np.float64)

    # Save to npz file
    output_file = OUTPUT_DIR / "cec2008_data_dim1000.npz"
    np.savez_compressed(output_file, **data)
    print(f"\nSaved data to {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")

    # Create __init__.py
    init_file = OUTPUT_DIR / "__init__.py"
    init_file.write_text("# CEC 2008 benchmark data\n")
    print(f"Created {init_file}")


if __name__ == "__main__":
    main()
