#!/usr/bin/env python
"""Generate CEC 2010 benchmark data files.

CEC 2010 is a large-scale optimization benchmark with 1000 dimensions and
partial separability (groups of 50 variables).

This script generates:
- Shift vectors for all 20 functions (F1-F20)
- Permutation indices for partial separability
- Rotation matrices (50x50) for the separable groups

Usage:
    python scripts/create_cec2010_data.py
"""

from pathlib import Path

import numpy as np
from scipy.stats import ortho_group

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data-packages" / "surfaces-cec-data" / "src" / "surfaces_cec_data" / "cec2010"


def generate_shift_vector(dim: int, bounds: tuple, seed: int) -> np.ndarray:
    """Generate a random shift vector within bounds."""
    np.random.seed(seed)
    lb, ub = bounds
    # Place optimum in inner 80% of search space
    margin = 0.1 * (ub - lb)
    return np.random.uniform(lb + margin, ub - margin, dim)


def generate_permutation(dim: int, seed: int) -> np.ndarray:
    """Generate a random permutation of indices."""
    np.random.seed(seed)
    return np.random.permutation(dim)


def generate_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    """Generate a random orthogonal matrix."""
    np.random.seed(seed)
    return ortho_group.rvs(dim)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dim = 1000
    m = 50  # Group size for partial separability
    n_groups = dim // m  # 20 groups

    data = {}

    # Generate shift vectors for all functions
    bounds_map = {
        1: (-100, 100),  # Elliptic
        2: (-5, 5),      # Rastrigin
        3: (-32, 32),    # Ackley
        4: (-100, 100),  # Elliptic
        5: (-5, 5),      # Rastrigin
        6: (-32, 32),    # Ackley
        7: (-100, 100),  # Schwefel
        8: (-100, 100),  # Elliptic
        9: (-5, 5),      # Rastrigin
        10: (-32, 32),   # Ackley
        11: (-100, 100), # Schwefel
        12: (-100, 100), # Rosenbrock
        13: (-600, 600), # Griewank
        14: (-100, 100), # Schwefel
        15: (-100, 100), # Rosenbrock
        16: (-5, 5),     # Rastrigin
        17: (-32, 32),   # Ackley
        18: (-600, 600), # Griewank
        19: (-5, 5),     # Composition 1
        20: (-5, 5),     # Composition 2
    }

    print("Generating CEC 2010 data files...")

    for func_id in range(1, 21):
        bounds = bounds_map.get(func_id, (-100, 100))
        print(f"  F{func_id}: shift vector (bounds={bounds})")
        shift = generate_shift_vector(dim, bounds, seed=2010 * 100 + func_id)
        data[f"shift_{func_id}"] = shift.astype(np.float64)

    # Generate permutation indices for partial separability
    print("  Generating permutation indices...")
    for func_id in range(1, 21):
        perm = generate_permutation(dim, seed=2010 * 200 + func_id)
        data[f"permutation_{func_id}"] = perm.astype(np.int32)

    # Generate small rotation matrices (50x50) for each group
    # This is much smaller than full 1000x1000 rotation
    print(f"  Generating rotation matrices ({m}x{m} per group)...")
    for func_id in range(1, 21):
        for group_id in range(n_groups):
            R = generate_rotation_matrix(m, seed=2010 * 1000 + func_id * 100 + group_id)
            data[f"rotation_{func_id}_g{group_id}"] = R.astype(np.float64)

    # Save to npz file
    output_file = OUTPUT_DIR / "cec2010_data_dim1000.npz"
    np.savez_compressed(output_file, **data)
    print(f"\nSaved data to {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024 * 1024):.2f} MB")

    # Create __init__.py
    init_file = OUTPUT_DIR / "__init__.py"
    init_file.write_text("# CEC 2010 benchmark data\n")
    print(f"Created {init_file}")


if __name__ == "__main__":
    main()
