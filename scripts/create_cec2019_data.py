#!/usr/bin/env python
"""Generate CEC 2019 benchmark data files.

CEC 2019 is the 100-Digit Challenge with 10 functions:
- F1: Storn's Chebyshev (D=9)
- F2: Inverse Hilbert (D=16)
- F3: Lennard-Jones (D=18)
- F4-F10: Shifted/Rotated functions (D=10)

This script generates shift vectors and rotation matrices.

Usage:
    python scripts/create_cec2019_data.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import ortho_group

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data-packages" / "surfaces-cec-data" / "src" / "surfaces_cec_data" / "cec2019"

# CEC 2019 has fixed dimensions per function
FUNC_DIMS = {
    1: 9,   # Storn's Chebyshev
    2: 16,  # Inverse Hilbert
    3: 18,  # Lennard-Jones
    4: 10,  # Rastrigin
    5: 10,  # Griewank
    6: 10,  # Weierstrass
    7: 10,  # Schwefel
    8: 10,  # Expanded Schaffer
    9: 10,  # Happy Cat
    10: 10, # Ackley
}


def generate_shift_vector(dim: int, bounds: tuple, seed: int) -> np.ndarray:
    """Generate a random shift vector within bounds."""
    np.random.seed(seed)
    lb, ub = bounds
    margin = 0.1 * (ub - lb)
    return np.random.uniform(lb + margin, ub - margin, dim)


def generate_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix."""
    np.random.seed(seed)
    return ortho_group.rvs(dim)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating CEC 2019 data files...")

    # For CEC 2019, we create one file per dimension since functions have different dims
    # But for simplicity, we'll create data for all functions in one file per unique dim

    all_data = {}

    for func_id, dim in FUNC_DIMS.items():
        seed_base = 2019 * 100 + func_id * 10

        # Define bounds based on function type
        if func_id == 1:  # Chebyshev
            bounds = (-8192, 8192)
        elif func_id == 2:  # Hilbert
            bounds = (-16384, 16384)
        elif func_id == 3:  # Lennard-Jones
            bounds = (-4, 4)
        else:  # F4-F10
            bounds = (-100, 100)

        # Shift vector
        shift = generate_shift_vector(dim, bounds, seed=seed_base)
        all_data[f"shift_{func_id}"] = shift.astype(np.float64)
        print(f"  F{func_id}: dim={dim}, bounds={bounds}")

        # Rotation matrix (for F4-F10)
        if func_id >= 4:
            R = generate_rotation_matrix(dim, seed=seed_base + 1000)
            all_data[f"rotation_{func_id}"] = R.astype(np.float64)

    # Save all data to a single file
    output_file = OUTPUT_DIR / "cec2019_data.npz"
    np.savez_compressed(output_file, **all_data)
    print(f"\nSaved {output_file}")
    print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")

    # Create __init__.py
    init_file = OUTPUT_DIR / "__init__.py"
    init_file.write_text("# CEC 2019 benchmark data\n")
    print(f"Created {init_file}")


if __name__ == "__main__":
    main()
