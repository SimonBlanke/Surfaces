# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

"""Pre-defined function presets for standardized optimizer testing."""

# =============================================================================
# Algebraic Functions (always available - numpy only)
# =============================================================================
from ..test_functions.algebraic import (
    # 1D
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    # ND
    GriewankFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    RastriginFunction,
    RosenbrockFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    SphereFunction,
    StyblinskiTangFunction,
    ThreeHumpCamelFunction,
)
from ..test_functions.bbob import (
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    BentCigar,
    BuecheRastrigin,
    DifferentPowers,
    Discus,
    # High Conditioning & Unimodal (f10-f14)
    EllipsoidalRotated,
    EllipsoidalSeparable,
    Gallagher21,
    Gallagher101,
    GriewankRosenbrock,
    Katsuura,
    LinearSlope,
    LunacekBiRastrigin,
    # Multimodal with Adequate Global Structure (f15-f19)
    RastriginRotated,
    RastriginSeparable,
    RosenbrockOriginal,
    RosenbrockRotated,
    SchaffersF7,
    SchaffersF7Ill,
    # Multimodal with Weak Global Structure (f20-f24)
    Schwefel,
    SharpRidge,
    StepEllipsoidal,
    Weierstrass,
)

# =============================================================================
# BBOB Functions (always available - numpy only)
# =============================================================================
from ..test_functions.bbob import (
    # Separable (f1-f5)
    Sphere as BBOB_Sphere,
)

# =============================================================================
# CEC Functions (always available - numpy only)
# =============================================================================
from ..test_functions.cec.cec2014 import (
    # Composition
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
    # Hybrid
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
    RotatedBentCigar,
    RotatedDiscus,
    # Unimodal
    RotatedHighConditionedElliptic,
    ShiftedRastrigin,
    ShiftedRotatedAckley,
    ShiftedRotatedExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScafferF6,
    ShiftedRotatedGriewank,
    ShiftedRotatedHappyCat,
    ShiftedRotatedHGBat,
    ShiftedRotatedKatsuura,
    ShiftedRotatedWeierstrass,
    ShiftedSchwefel,
)
from ..test_functions.cec.cec2014 import (
    ShiftedRotatedRastrigin as CEC2014_ShiftedRotatedRastrigin,
)
from ..test_functions.cec.cec2014 import (
    # Multimodal
    ShiftedRotatedRosenbrock as CEC2014_ShiftedRotatedRosenbrock,
)
from ..test_functions.cec.cec2014 import (
    ShiftedRotatedSchwefel as CEC2014_ShiftedRotatedSchwefel,
)
from ..test_functions.cec.cec2017 import (
    ShiftedRotatedBentCigar,
    ShiftedRotatedLevy,
    ShiftedRotatedLunacekBiRastrigin,
    ShiftedRotatedNonContRastrigin,
    ShiftedRotatedSchafferF7,
    ShiftedRotatedSumDiffPow,
    ShiftedRotatedZakharov,
)
from ..test_functions.cec.cec2017 import (
    ShiftedRotatedRastrigin as CEC2017_ShiftedRotatedRastrigin,
)
from ..test_functions.cec.cec2017 import (
    ShiftedRotatedRosenbrock as CEC2017_ShiftedRotatedRosenbrock,
)
from ..test_functions.cec.cec2017 import (
    ShiftedRotatedSchwefel as CEC2017_ShiftedRotatedSchwefel,
)

# =============================================================================
# Engineering Functions (always available - numpy only)
# =============================================================================
from ..test_functions.engineering import (
    CantileverBeamFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    ThreeBarTrussFunction,
    WeldedBeamFunction,
)

# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

quick = [
    SphereFunction,  # Simplest convex function, baseline
    AckleyFunction,  # Multimodal with global structure
    RosenbrockFunction,  # Valley/banana shape, tests gradient following
    RastriginFunction,  # Highly multimodal, periodic local optima
    GriewankFunction,  # Mixed local/global structure
]
"""Fast sanity check (5 functions).

Purpose: Verify optimizer runs without crashing.
Use case: CI smoke tests, development iteration.
Selection: Well-known, diverse characteristics, fast to evaluate.
"""

standard = [
    # Unimodal (baseline)
    SphereFunction,
    RosenbrockFunction,
    # Multimodal - global structure
    AckleyFunction,
    GriewankFunction,
    # Multimodal - local structure
    RastriginFunction,
    StyblinskiTangFunction,
    # 2D classics (well-visualized, well-understood)
    BealeFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    LeviFunctionN13,
    # Challenging landscapes
    EggholderFunction,
    SchafferFunctionN2,
    DropWaveFunction,
    # Deceptive/tricky
    EasomFunction,
    CrossInTrayFunction,
]
"""Academic comparison (15 functions).

Purpose: Balanced test for publication-quality benchmarking.
Use case: Papers, algorithm comparison studies.
Selection: Classic functions, diverse landscape types, well-studied.
"""

algebraic_2d = [
    AckleyFunction,
    BealeFunction,
    BoothFunction,
    BukinFunctionN6,
    CrossInTrayFunction,
    DropWaveFunction,
    EasomFunction,
    EggholderFunction,
    GoldsteinPriceFunction,
    HimmelblausFunction,
    HölderTableFunction,
    LangermannFunction,
    LeviFunctionN13,
    MatyasFunction,
    McCormickFunction,
    SchafferFunctionN2,
    SimionescuFunction,
    ThreeHumpCamelFunction,
]
"""All 2D algebraic functions (18 functions).

Purpose: Complete 2D function coverage.
Use case: Visualization studies, 2D algorithm behavior analysis.
"""

algebraic_nd = [
    SphereFunction,
    RosenbrockFunction,
    RastriginFunction,
    GriewankFunction,
    StyblinskiTangFunction,
]
"""N-dimensional scalable functions (5 functions).

Purpose: Dimensionality and scalability studies.
Use case: Testing how algorithms scale with dimension.
Note: Requires n_dim parameter when instantiating.
"""

bbob = [
    # Separable (f1-f5)
    BBOB_Sphere,
    EllipsoidalSeparable,
    RastriginSeparable,
    BuecheRastrigin,
    LinearSlope,
    # Low/Moderate Conditioning (f6-f9)
    AttractiveSector,
    StepEllipsoidal,
    RosenbrockOriginal,
    RosenbrockRotated,
    # High Conditioning & Unimodal (f10-f14)
    EllipsoidalRotated,
    Discus,
    BentCigar,
    SharpRidge,
    DifferentPowers,
    # Multimodal with Adequate Global Structure (f15-f19)
    RastriginRotated,
    Weierstrass,
    SchaffersF7,
    SchaffersF7Ill,
    GriewankRosenbrock,
    # Multimodal with Weak Global Structure (f20-f24)
    Schwefel,
    Gallagher101,
    Gallagher21,
    Katsuura,
    LunacekBiRastrigin,
]
"""Full COCO/BBOB benchmark (24 functions).

Purpose: Standardized comparison per COCO platform.
Use case: GECCO papers, EMO conference, algorithm development.
Reference: Hansen et al. (2009) BBOB function definitions.
Note: Requires n_dim parameter when instantiating.
"""

cec2014 = [
    # Unimodal (F1-F3)
    RotatedHighConditionedElliptic,
    RotatedBentCigar,
    RotatedDiscus,
    # Simple Multimodal (F4-F16)
    CEC2014_ShiftedRotatedRosenbrock,
    ShiftedRotatedAckley,
    ShiftedRotatedWeierstrass,
    ShiftedRotatedGriewank,
    ShiftedRastrigin,
    CEC2014_ShiftedRotatedRastrigin,
    ShiftedSchwefel,
    CEC2014_ShiftedRotatedSchwefel,
    ShiftedRotatedKatsuura,
    ShiftedRotatedHappyCat,
    ShiftedRotatedHGBat,
    ShiftedRotatedExpandedGriewankRosenbrock,
    ShiftedRotatedExpandedScafferF6,
    # Hybrid (F17-F22)
    HybridFunction1,
    HybridFunction2,
    HybridFunction3,
    HybridFunction4,
    HybridFunction5,
    HybridFunction6,
    # Composition (F23-F30)
    CompositionFunction1,
    CompositionFunction2,
    CompositionFunction3,
    CompositionFunction4,
    CompositionFunction5,
    CompositionFunction6,
    CompositionFunction7,
    CompositionFunction8,
]
"""CEC 2014 competition functions (30 functions).

Purpose: IEEE CEC 2014 competition benchmark.
Use case: Evolutionary computation papers, competition-style comparison.
Reference: Liang et al. (2013) CEC 2014 problem definitions.
Note: Requires n_dim parameter when instantiating.
"""

cec2017 = [
    ShiftedRotatedBentCigar,
    ShiftedRotatedSumDiffPow,
    ShiftedRotatedZakharov,
    CEC2017_ShiftedRotatedRosenbrock,
    CEC2017_ShiftedRotatedRastrigin,
    ShiftedRotatedSchafferF7,
    ShiftedRotatedLunacekBiRastrigin,
    ShiftedRotatedNonContRastrigin,
    ShiftedRotatedLevy,
    CEC2017_ShiftedRotatedSchwefel,
]
"""CEC 2017 simple functions (10 functions).

Purpose: IEEE CEC 2017 competition benchmark (simple category).
Use case: Recent competition benchmarks.
Reference: Awad et al. (2016) CEC 2017 problem definitions.
Note: Requires n_dim parameter when instantiating.
"""

engineering = [
    ThreeBarTrussFunction,
    WeldedBeamFunction,
    PressureVesselFunction,
    TensionCompressionSpringFunction,
    CantileverBeamFunction,
]
"""Constrained engineering problems (5 functions).

Purpose: Test constraint handling capabilities.
Use case: Constrained optimization algorithm evaluation.
Note: Functions use penalty methods; check is_feasible() for solutions.
"""
