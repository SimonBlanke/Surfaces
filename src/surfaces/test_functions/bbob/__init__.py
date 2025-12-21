# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""BBOB (Black-Box Optimization Benchmarking) Functions.

This module provides the 24 noiseless benchmark functions from the BBOB test suite,
part of the COCO (Comparing Continuous Optimizers) platform. These functions are
widely used in the evolutionary computation community for algorithm comparison.

Functions are organized into five categories:
- Separable (f1-f5): Functions that can be optimized dimension-by-dimension
- Low/Moderate Conditioning (f6-f9): Functions with condition numbers <= 10
- High Conditioning & Unimodal (f10-f14): Ill-conditioned unimodal functions
- Multimodal with Adequate Global Structure (f15-f19): Multimodal with global pattern
- Multimodal with Weak Global Structure (f20-f24): Highly deceptive multimodal

Key Features:
- Search domain: [-5, 5]^D
- Instance-based: Each instance has different random transformations
- Reproducible: Results depend on (func_id, n_dim, instance) seed

Reference:
    Hansen, N., Finck, S., Ros, R., & Auger, A. (2009).
    Real-parameter black-box optimization benchmarking 2009: Noiseless functions definitions.
    Technical Report RR-6829, INRIA.
"""

from ._base_bbob import BBOBFunction
from .high_conditioning import (
    BentCigar,
    DifferentPowers,
    Discus,
    EllipsoidalRotated,
    SharpRidge,
)
from .low_conditioning import (
    AttractiveSector,
    RosenbrockOriginal,
    RosenbrockRotated,
    StepEllipsoidal,
)
from .multimodal_adequate import (
    GriewankRosenbrock,
    RastriginRotated,
    SchaffersF7,
    SchaffersF7Ill,
    Weierstrass,
)
from .multimodal_weak import (
    Gallagher21,
    Gallagher101,
    Katsuura,
    LunacekBiRastrigin,
    Schwefel,
)
from .separable import (
    BuecheRastrigin,
    EllipsoidalSeparable,
    LinearSlope,
    RastriginSeparable,
    Sphere,
)

__all__ = [
    "BBOBFunction",
    # Separable (f1-f5)
    "Sphere",
    "EllipsoidalSeparable",
    "RastriginSeparable",
    "BuecheRastrigin",
    "LinearSlope",
    # Low/Moderate Conditioning (f6-f9)
    "AttractiveSector",
    "StepEllipsoidal",
    "RosenbrockOriginal",
    "RosenbrockRotated",
    # High Conditioning & Unimodal (f10-f14)
    "EllipsoidalRotated",
    "Discus",
    "BentCigar",
    "SharpRidge",
    "DifferentPowers",
    # Multimodal with Adequate Global Structure (f15-f19)
    "RastriginRotated",
    "Weierstrass",
    "SchaffersF7",
    "SchaffersF7Ill",
    "GriewankRosenbrock",
    # Multimodal with Weak Global Structure (f20-f24)
    "Schwefel",
    "Gallagher101",
    "Gallagher21",
    "Katsuura",
    "LunacekBiRastrigin",
]

# Function ID to class mapping
BBOB_FUNCTIONS = {
    1: Sphere,
    2: EllipsoidalSeparable,
    3: RastriginSeparable,
    4: BuecheRastrigin,
    5: LinearSlope,
    6: AttractiveSector,
    7: StepEllipsoidal,
    8: RosenbrockOriginal,
    9: RosenbrockRotated,
    10: EllipsoidalRotated,
    11: Discus,
    12: BentCigar,
    13: SharpRidge,
    14: DifferentPowers,
    15: RastriginRotated,
    16: Weierstrass,
    17: SchaffersF7,
    18: SchaffersF7Ill,
    19: GriewankRosenbrock,
    20: Schwefel,
    21: Gallagher101,
    22: Gallagher21,
    23: Katsuura,
    24: LunacekBiRastrigin,
}
