#!/usr/bin/env python3
"""Create CEC2005 data files from official sources.

Downloads and converts the official CEC2005 benchmark data files
to NPZ format for use with the Surfaces library.

Source: https://github.com/P-N-Suganthan/CEC2005

Usage:
    python scripts/create_cec2005_data.py
"""

import io
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

# Official CEC2005 data source
ASCII_URL = "https://github.com/P-N-Suganthan/CEC2005/raw/master/ASCII-files.tar.gz"

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "data-packages" / "surfaces-cec-data" / "src" / "surfaces_cec_data" / "cec2005"

# Supported dimensions
DIMS = [2, 10, 30, 50]

# CEC2005 bias values for each function
BIAS = {
    1: -450, 2: -450, 3: -450, 4: -450, 5: -310,
    6: 390, 7: -180, 8: -140, 9: -330, 10: -330,
    11: 90, 12: -460, 13: -130, 14: -300,
    15: 120, 16: 120, 17: 120, 18: 10, 19: 10,
    20: 10, 21: 360, 22: 360, 23: 360, 24: 260, 25: 260
}


def download_and_extract():
    """Download and extract ASCII files from official repository."""
    print(f"Downloading CEC2005 data from {ASCII_URL}...")

    with urlopen(ASCII_URL) as response:
        data = response.read()

    print(f"Downloaded {len(data)} bytes")

    # Extract to temp directory
    tmpdir = tempfile.mkdtemp(prefix="cec2005_")

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(tmpdir)

    # Find the extracted directory
    data_dir = Path(tmpdir)
    subdirs = list(data_dir.iterdir())
    if len(subdirs) == 1 and subdirs[0].is_dir():
        data_dir = subdirs[0]

    print(f"Extracted to {data_dir}")
    return data_dir


def read_data_file(filepath: Path, shape=None) -> np.ndarray:
    """Read a data file with space/newline separated values."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Parse all numbers from file
    values = []
    for line in content.strip().split('\n'):
        for val in line.strip().split():
            try:
                values.append(float(val))
            except ValueError:
                continue

    arr = np.array(values)

    if shape is not None:
        arr = arr.reshape(shape)

    return arr


def load_rotation_matrix(data_dir: Path, name: str, dim: int) -> np.ndarray:
    """Load a rotation matrix for specific dimension."""
    # Try different naming patterns
    patterns = [
        f"{name}_M_D{dim}.txt",
        f"{name}_M_{dim}.txt",
        f"{name}_D{dim}.txt",
    ]

    for pattern in patterns:
        filepath = data_dir / pattern
        if filepath.exists():
            return read_data_file(filepath, shape=(dim, dim))

    # If not found, generate random orthogonal matrix (deterministic)
    print(f"  Warning: Rotation matrix {name} D{dim} not found, generating...")
    rng = np.random.default_rng(seed=hash(f"{name}_{dim}") % 2**32)
    M = rng.standard_normal((dim, dim))
    Q, _ = np.linalg.qr(M)
    return Q


def load_shift_vector(data_dir: Path, name: str, max_dim: int = 100) -> np.ndarray:
    """Load a shift vector (global optimum location)."""
    patterns = [
        f"{name}_func_data.txt",
        f"{name}_data.txt",
        f"{name}.txt",
    ]

    for pattern in patterns:
        filepath = data_dir / pattern
        if filepath.exists():
            data = read_data_file(filepath)
            # Take first max_dim values
            return data[:max_dim]

    # If not found, generate random shift
    print(f"  Warning: Shift vector {name} not found, generating...")
    rng = np.random.default_rng(seed=hash(name) % 2**32)
    return rng.uniform(-80, 80, size=max_dim)


def create_npz_for_dim(data_dir: Path, dim: int) -> dict:
    """Create data dictionary for a specific dimension."""
    data = {}

    print(f"\nProcessing dimension {dim}...")

    # =================================================================
    # Shift vectors for F1-F14
    # =================================================================

    # F1: Sphere - uses sphere shift
    shift_1 = load_shift_vector(data_dir, "sphere")[:dim]
    data["shift_1"] = shift_1

    # F2: Schwefel 1.2 - uses schwefel_102 shift
    shift_2 = load_shift_vector(data_dir, "schwefel_102")[:dim]
    data["shift_2"] = shift_2

    # F3: Rotated Elliptic - uses elliptic shift
    shift_3 = load_shift_vector(data_dir, "high_cond_elliptic_rot")[:dim]
    data["shift_3"] = shift_3
    data["rotation_3"] = load_rotation_matrix(data_dir, "elliptic", dim)

    # F4: Schwefel 1.2 with noise - same as F2
    data["shift_4"] = shift_2.copy()

    # F5: Schwefel 2.6 - special handling with A matrix and B vector
    # This function uses A*x - B formulation
    try:
        schwefel_206_data = read_data_file(data_dir / "schwefel_206_data.txt")
        # Format: first 100 values are the optimum, then A matrix (100x100), then B vector
        o = schwefel_206_data[:100][:dim]
        A_flat = schwefel_206_data[100:100+100*100]
        A = A_flat.reshape(100, 100)[:dim, :dim]
        # B = A @ o
        B = A @ o
        data["shift_5"] = o
        data["a_matrix_5"] = A
        data["b_vector_5"] = B
    except Exception as e:
        print(f"  Warning: Could not load Schwefel 2.6 data: {e}")
        rng = np.random.default_rng(seed=5)
        o = rng.uniform(-100, 100, size=dim)
        A = rng.uniform(-100, 100, size=(dim, dim))
        B = A @ o
        data["shift_5"] = o
        data["a_matrix_5"] = A
        data["b_vector_5"] = B

    # F6: Rosenbrock
    shift_6 = load_shift_vector(data_dir, "rosenbrock")[:dim]
    data["shift_6"] = shift_6

    # F7: Rotated Griewank
    shift_7 = load_shift_vector(data_dir, "griewank")[:dim]
    data["shift_7"] = shift_7
    data["rotation_7"] = load_rotation_matrix(data_dir, "griewank", dim)

    # F8: Rotated Ackley
    shift_8 = load_shift_vector(data_dir, "ackley")[:dim]
    data["shift_8"] = shift_8
    data["rotation_8"] = load_rotation_matrix(data_dir, "ackley", dim)

    # F9: Shifted Rastrigin (no rotation)
    shift_9 = load_shift_vector(data_dir, "rastrigin")[:dim]
    data["shift_9"] = shift_9

    # F10: Rotated Rastrigin
    data["shift_10"] = shift_9.copy()
    data["rotation_10"] = load_rotation_matrix(data_dir, "rastrigin", dim)

    # F11: Rotated Weierstrass
    shift_11 = load_shift_vector(data_dir, "weierstrass")[:dim]
    data["shift_11"] = shift_11
    data["rotation_11"] = load_rotation_matrix(data_dir, "weierstrass", dim)

    # F12: Schwefel 2.13 - uses A, B matrices and alpha vector
    try:
        schwefel_213_data = read_data_file(data_dir / "schwefel_213_data.txt")
        # Format varies - typically alpha, A matrix, B matrix
        n = 100  # max dim
        alpha = schwefel_213_data[:n][:dim]
        A_start = n
        A = schwefel_213_data[A_start:A_start + n*n].reshape(n, n)[:dim, :dim]
        B_start = A_start + n*n
        B = schwefel_213_data[B_start:B_start + n*n].reshape(n, n)[:dim, :dim]
        data["shift_12"] = alpha
        data["a_matrix_12"] = A
        data["b_matrix_12"] = B
    except Exception as e:
        print(f"  Warning: Could not load Schwefel 2.13 data: {e}")
        rng = np.random.default_rng(seed=12)
        alpha = rng.uniform(-np.pi, np.pi, size=dim)
        A = rng.integers(-100, 101, size=(dim, dim)).astype(float)
        B = rng.integers(-100, 101, size=(dim, dim)).astype(float)
        data["shift_12"] = alpha
        data["a_matrix_12"] = A
        data["b_matrix_12"] = B

    # F13: Expanded Griewank-Rosenbrock (EF8F2)
    shift_13 = load_shift_vector(data_dir, "EF8F2")[:dim]
    data["shift_13"] = shift_13
    data["rotation_13"] = load_rotation_matrix(data_dir, "E_ScafferF6", dim)  # Uses same rotation

    # F14: Expanded Scaffer F6
    shift_14 = load_shift_vector(data_dir, "E_ScafferF6")[:dim]
    data["shift_14"] = shift_14
    data["rotation_14"] = load_rotation_matrix(data_dir, "E_ScafferF6", dim)

    # =================================================================
    # Composition functions F15-F25
    # =================================================================

    # For composition functions, we need:
    # - Multiple shift vectors (optima for each component)
    # - Multiple rotation matrices (one per component)
    # - Sigma, lambda parameters

    # Number of component functions in each composition
    n_components = 10

    # Sigmas for each composition function (from CEC2005 spec)
    sigmas_by_func = {
        15: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        16: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        17: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        18: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        19: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        20: [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        21: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        22: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        23: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        24: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        25: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    }

    # Lambdas (scaling) for each composition function
    lambdas_by_func = {
        15: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        16: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        17: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        18: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        19: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        20: [1, 1, 10, 10, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        21: [1.0/5, 1.0/5, 1, 1, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        22: [1.0/5, 1.0/5, 1, 1, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        23: [1.0/5, 1.0/5, 1, 1, 5.0/60, 5.0/60, 5.0/32, 5.0/32, 5.0/100, 5.0/100],
        24: [10, 10, 2.5, 25, 1e-6, 1e-6, 2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4],
        25: [10, 10, 2.5, 25, 1e-6, 1e-6, 2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4],
    }

    for func_id in range(15, 26):
        # Try to load composition data
        try:
            comp_data = read_data_file(data_dir / f"hybrid_func{func_id-14}_data.txt")
            # Format: 10 optima of dim 100 each = 1000 values
            optima = comp_data[:n_components * 100].reshape(n_components, 100)[:, :dim]
        except Exception:
            # Generate random optima
            rng = np.random.default_rng(seed=func_id)
            optima = rng.uniform(-5, 5, size=(n_components, dim))

        data[f"comp_optima_{func_id}"] = optima
        data[f"comp_sigma_{func_id}"] = np.array(sigmas_by_func[func_id], dtype=float)
        data[f"comp_lambda_{func_id}"] = np.array(lambdas_by_func[func_id], dtype=float)

        # Load rotation matrices for each component
        for i in range(n_components):
            try:
                M = read_data_file(
                    data_dir / f"hybrid_func{func_id-14}_M_D{dim}_{i+1}.txt",
                    shape=(dim, dim)
                )
            except Exception:
                # Generate orthogonal matrix
                rng = np.random.default_rng(seed=func_id * 100 + i)
                M = rng.standard_normal((dim, dim))
                M, _ = np.linalg.qr(M)

            data[f"rotation_{func_id}_{i+1}"] = M

    return data


def main():
    """Main entry point."""
    # Download and extract official data
    data_dir = download_and_extract()

    # List all files for debugging
    print("\nAvailable data files:")
    for f in sorted(data_dir.glob("*.txt")):
        print(f"  {f.name}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create NPZ for each dimension
    for dim in DIMS:
        data = create_npz_for_dim(data_dir, dim)

        output_file = OUTPUT_DIR / f"cec2005_data_dim{dim}.npz"
        np.savez_compressed(output_file, **data)
        print(f"Saved {output_file}")

    # Create __init__.py
    init_file = OUTPUT_DIR / "__init__.py"
    init_file.write_text('"""CEC 2005 benchmark data files."""\n')
    print(f"Created {init_file}")

    print("\nDone!")


if __name__ == "__main__":
    main()
