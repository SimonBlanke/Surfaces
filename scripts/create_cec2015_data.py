#!/usr/bin/env python
"""Generate CEC 2015 benchmark data files.

CEC 2015 is a learning-based optimization benchmark with 15 functions
supporting dimensions 10 and 30.

This script generates:
- Shift vectors for all 15 functions
- Rotation matrices for rotated functions
- Shuffle indices for hybrid functions

Usage:
    python scripts/create_cec2015_data.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import ortho_group

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data-packages" / "surfaces-cec-data" / "src" / "surfaces_cec_data" / "cec2015"

# Supported dimensions
DIMS = [10, 30]


def generate_shift_vector(dim: int, bounds: tuple, seed: int) -> np.ndarray:
    """Generate a random shift vector within bounds."""
    np.random.seed(seed)
    lb, ub = bounds
    # Place optimum in inner 80% of search space
    margin = 0.1 * (ub - lb)
    return np.random.uniform(lb + margin, ub - margin, dim)


def generate_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix."""
    np.random.seed(seed)
    return ortho_group.rvs(dim)


def generate_shuffle_indices(dim: int, seed: int) -> np.ndarray:
    """Generate shuffle indices for hybrid functions."""
    np.random.seed(seed)
    return np.random.permutation(dim)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating CEC 2015 data files...")

    for dim in DIMS:
        print(f"\nDimension {dim}:")
        data = {}

        # Generate shift vectors and rotation matrices for all 15 functions
        for func_id in range(1, 16):
            seed_base = 2015 * 100 + func_id * 10 + dim

            # Shift vector
            shift = generate_shift_vector(dim, (-80, 80), seed=seed_base)
            data[f"shift_{func_id}"] = shift.astype(np.float64)
            print(f"  F{func_id}: shift vector")

            # Rotation matrix
            R = generate_rotation_matrix(dim, seed=seed_base + 1000)
            data[f"rotation_{func_id}"] = R.astype(np.float64)

        # Generate shuffle indices for hybrid functions (F10-F12)
        for func_id in range(10, 13):
            seed = 2015 * 200 + func_id * 10 + dim
            shuffle = generate_shuffle_indices(dim, seed)
            data[f"shuffle_{func_id}"] = shuffle.astype(np.int32)
            print(f"  F{func_id}: shuffle indices")

        # Composition function data (F13-F15)
        # Each composition needs multiple optima and rotation matrices
        n_components_map = {13: 5, 14: 3, 15: 5}

        for func_id in [13, 14, 15]:
            n_comp = n_components_map[func_id]
            seed_base = 2015 * 300 + func_id * 100 + dim

            # Generate optima for each component
            optima = np.zeros((n_comp, dim))
            for i in range(n_comp):
                np.random.seed(seed_base + i)
                optima[i] = np.random.uniform(-80, 80, dim)
            data[f"comp_optima_{func_id}"] = optima.astype(np.float64)

            # Generate rotation matrices for each component
            for i in range(n_comp):
                R = generate_rotation_matrix(dim, seed=seed_base + 1000 + i)
                data[f"rotation_{func_id}_{i+1}"] = R.astype(np.float64)

            print(f"  F{func_id}: composition ({n_comp} components)")

        # Save to npz file
        output_file = OUTPUT_DIR / f"cec2015_data_dim{dim}.npz"
        np.savez_compressed(output_file, **data)
        print(f"Saved {output_file}")
        print(f"File size: {output_file.stat().st_size / 1024:.2f} KB")

    # Create __init__.py
    init_file = OUTPUT_DIR / "__init__.py"
    init_file.write_text("# CEC 2015 benchmark data\n")
    print(f"\nCreated {init_file}")


if __name__ == "__main__":
    main()
