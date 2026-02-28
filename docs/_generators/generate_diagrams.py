#!/usr/bin/env python3
"""Generate diagrams and overview content for documentation pages.

This generator creates Mermaid diagrams and introductory RST content
for all overview pages. Content is auto-generated from source code
to stay in sync with the actual module structure.

Output Files
------------
docs/source/_generated/diagrams/
    test_functions_overview.rst
    algebraic_overview.rst
    bbob_overview.rst
    cec_overview.rst
    ml_overview.rst
    engineering_overview.rst
    presets_overview.rst
    noise_overview.rst
    visualization_overview.rst
    base_classes_overview.rst
    user_guide_overview.rst
    examples_overview.rst

Usage
-----
    python -m docs._generators.generate_diagrams
"""

from . import count_by_category, get_total_count
from .config import DIAGRAMS_DIR


def generate_test_functions_overview() -> str:
    """Generate overview for Test Functions index page."""
    counts = count_by_category()
    total = get_total_count()

    n_algebraic = (
        counts.get("algebraic_1d", 0)
        + counts.get("algebraic_2d", 0)
        + counts.get("algebraic_nd", 0)
    )
    n_ml = sum(v for k, v in counts.items() if k.startswith("ml_"))
    n_eng = counts.get("engineering", 0)
    n_bbob = counts.get("bbob", 0)
    n_cec = counts.get("cec", 0)

    return f"""Test functions are callable objects that simulate optimization landscapes.
They provide a controlled environment for testing and benchmarking optimization algorithms.

Surfaces provides **{total} test functions** across five categories, each designed for different testing scenarios.

.. mermaid::

    flowchart LR
        TF[Test Functions<br/><b>{total} total</b>]

        TF --> ALG[Algebraic<br/>{n_algebraic} functions]
        TF --> BBOB[BBOB<br/>{n_bbob} functions]
        TF --> CEC[CEC<br/>{n_cec} functions]
        TF --> ML[Machine Learning<br/>{n_ml} functions]
        TF --> ENG[Engineering<br/>{n_eng} functions]

        ALG --> ALG1[1D: {counts.get('algebraic_1d', 0)}]
        ALG --> ALG2[2D: {counts.get('algebraic_2d', 0)}]
        ALG --> ALGN[N-D: {counts.get('algebraic_nd', 0)}]

        style TF fill:#1A1854,color:#fff
        style ALG fill:#3D3A7A,color:#fff
        style BBOB fill:#3D3A7A,color:#fff
        style CEC fill:#3D3A7A,color:#fff
        style ML fill:#3D3A7A,color:#fff
        style ENG fill:#3D3A7A,color:#fff
"""


def generate_algebraic_overview() -> str:
    """Generate overview for Algebraic Functions page."""
    counts = count_by_category()
    n_1d = counts.get("algebraic_1d", 0)
    n_2d = counts.get("algebraic_2d", 0)
    n_nd = counts.get("algebraic_nd", 0)
    total = n_1d + n_2d + n_nd

    return f"""Algebraic functions are classic mathematical optimization benchmarks from the literature.
They have known global optima, making them ideal for validating optimizer correctness.

This category contains **{total} functions** organized by dimensionality.

.. mermaid::

    flowchart TD
        subgraph Algebraic Functions
            A[Algebraic<br/><b>{total} functions</b>]

            A --> D1[1D Functions<br/>{n_1d} functions]
            A --> D2[2D Functions<br/>{n_2d} functions]
            A --> DN[N-D Functions<br/>{n_nd} functions]

            D1 --> D1a[Single variable<br/>Simple landscapes]
            D2 --> D2a[Visualizable<br/>Surface plots]
            DN --> DNa[Scalable<br/>Any dimension]
        end

        style A fill:#1A1854,color:#fff
        style D1 fill:#5A7A8B,color:#fff
        style D2 fill:#5A7A8B,color:#fff
        style DN fill:#5A7A8B,color:#fff

**Key Properties:**

- Known global optima for validation
- LaTeX formulas available for documentation
- Support for noise injection
- Compatible with scipy.optimize
"""


def generate_bbob_overview() -> str:
    """Generate overview for BBOB Functions page."""
    # Get BBOB function counts by category (from __init__.py structure)
    try:
        import surfaces.test_functions.benchmark.bbob  # noqa: F401

        # Count based on BBOB function ID ranges
        # f1-f5: Separable, f6-f9: Low, f10-f14: High, f15-f19: Adequate, f20-f24: Weak
        n_sep = 5
        n_low = 4
        n_high = 5
        n_adequate = 5
        n_weak = 5
    except ImportError:
        n_sep = n_low = n_high = n_adequate = n_weak = 0

    total = n_sep + n_low + n_high + n_adequate + n_weak

    return f"""The BBOB (Black-Box Optimization Benchmarking) test suite is part of the COCO platform,
widely used in the evolutionary computation community for algorithm comparison.

Surfaces provides all **{total} noiseless BBOB functions** (f1-f24), organized into five categories based on their properties.

.. mermaid::

    flowchart TD
        BBOB[BBOB Suite<br/><b>{total} functions</b>]

        BBOB --> SEP[Separable<br/>f1-f5: {n_sep} functions]
        BBOB --> LOW[Low Conditioning<br/>f6-f9: {n_low} functions]
        BBOB --> HIGH[High Conditioning<br/>f10-f14: {n_high} functions]
        BBOB --> ADQ[Multimodal Adequate<br/>f15-f19: {n_adequate} functions]
        BBOB --> WEAK[Multimodal Weak<br/>f20-f24: {n_weak} functions]

        SEP --> SEPd[Dimension-by-dimension<br/>optimization possible]
        HIGH --> HIGHd[Ill-conditioned<br/>unimodal]
        WEAK --> WEAKd[Highly deceptive<br/>many local optima]

        style BBOB fill:#1A1854,color:#fff
        style SEP fill:#5A7A8B,color:#fff
        style LOW fill:#5A7A8B,color:#fff
        style HIGH fill:#5A7A8B,color:#fff
        style ADQ fill:#5A7A8B,color:#fff
        style WEAK fill:#5A7A8B,color:#fff

**Key Properties:**

- Search domain: [-5, 5]^D for all functions
- Instance-based: random transformations for each instance
- Scalable: works with any number of dimensions
- Reproducible: results depend on (func_id, n_dim, instance) seed
"""


def generate_cec_overview() -> str:
    """Generate overview for CEC Functions page."""
    try:
        from surfaces.test_functions.benchmark.cec import cec2013, cec2014, cec2017

        # Count excluding base classes
        n_2013 = len([x for x in cec2013.__all__ if not x.startswith("CEC20")])
        n_2014 = len([x for x in cec2014.__all__ if not x.startswith("CEC20")])
        n_2017 = len([x for x in cec2017.__all__ if not x.startswith("CEC20")])
    except (ImportError, AttributeError):
        n_2013 = 28
        n_2014 = 30
        n_2017 = 10

    total = n_2013 + n_2014 + n_2017

    return f"""The CEC (Congress on Evolutionary Computation) benchmark suites are competition standards
used for comparing optimization algorithms in academic research.

Surfaces provides **{total} CEC functions** from three competition years.

.. mermaid::

    flowchart TD
        CEC[CEC Benchmarks<br/><b>{total} functions</b>]

        CEC --> C13[CEC 2013<br/>{n_2013} functions]
        CEC --> C14[CEC 2014<br/>{n_2014} functions]
        CEC --> C17[CEC 2017<br/>{n_2017} functions]

        C13 --> C13a[Unimodal: F1-F5]
        C13 --> C13b[Multimodal: F6-F20]
        C13 --> C13c[Composition: F21-F28]

        C14 --> C14a[Unimodal + Multimodal]
        C14 --> C14b[Hybrid: F17-F22]
        C14 --> C14c[Composition: F23-F30]

        C17 --> C17a[Simple: F1-F10]

        style CEC fill:#1A1854,color:#fff
        style C13 fill:#5A7A8B,color:#fff
        style C14 fill:#5A7A8B,color:#fff
        style C17 fill:#5A7A8B,color:#fff

**Key Properties:**

- Shifted and rotated variants
- Composition functions combine multiple landscapes
- Standard dimensions: 10, 30, 50, 100
- Official evaluation criteria from competition specs
"""


def generate_ml_overview() -> str:
    """Generate overview for Machine Learning Functions page."""
    counts = count_by_category()
    n_tab_clf = counts.get("ml_tabular_classification", 0)
    n_tab_reg = counts.get("ml_tabular_regression", 0)
    n_img = counts.get("ml_image_classification", 0)
    n_ts_clf = counts.get("ml_timeseries_classification", 0)
    n_ts_fc = counts.get("ml_timeseries_forecasting", 0)

    n_tabular = n_tab_clf + n_tab_reg
    n_ts = n_ts_clf + n_ts_fc
    total = n_tabular + n_img + n_ts

    return f"""Machine learning functions represent hyperparameter optimization landscapes.
Each function wraps a scikit-learn model and evaluates hyperparameter configurations on real datasets.

Surfaces provides **{total} ML functions** across three data types.

.. mermaid::

    flowchart TD
        ML[Machine Learning<br/><b>{total} functions</b>]

        ML --> TAB[Tabular Data<br/>{n_tabular} functions]
        ML --> IMG[Image Data<br/>{n_img} functions]
        ML --> TS[Time Series<br/>{n_ts} functions]

        TAB --> TAB_CLF[Classification<br/>{n_tab_clf} models]
        TAB --> TAB_REG[Regression<br/>{n_tab_reg} models]

        IMG --> IMG_CLF[Classification<br/>{n_img} models]

        TS --> TS_CLF[Classification<br/>{n_ts_clf} models]
        TS --> TS_FC[Forecasting<br/>{n_ts_fc} models]

        style ML fill:#1A1854,color:#fff
        style TAB fill:#5A8B6A,color:#fff
        style IMG fill:#5A8B6A,color:#fff
        style TS fill:#5A8B6A,color:#fff

**Key Properties:**

- Real evaluation cost (actual model training)
- Stochastic landscapes (cross-validation variance)
- Practical hyperparameter ranges
- Built-in datasets (Iris, Wine, MNIST, etc.)
"""


def generate_engineering_overview() -> str:
    """Generate overview for Engineering Functions page."""
    counts = count_by_category()
    n_eng = counts.get("engineering", 0)

    return f"""Engineering functions represent real-world constrained design optimization problems.
They include inequality constraints that must be satisfied for feasible solutions.

Surfaces provides **{n_eng} engineering problems** from structural and mechanical design.

.. mermaid::

    flowchart TD
        ENG[Engineering Design<br/><b>{n_eng} functions</b>]

        ENG --> OBJ[Objective]
        ENG --> CON[Constraints]

        OBJ --> COST[Minimize Cost<br/>or Weight]

        CON --> STRESS[Stress Limits]
        CON --> DEFLECT[Deflection Limits]
        CON --> GEOM[Geometric Constraints]

        subgraph Problems
            P1[Pressure Vessel]
            P2[Welded Beam]
            P3[Tension Spring]
            P4[Three-Bar Truss]
            P5[Cantilever Beam]
        end

        ENG --> Problems

        style ENG fill:#1A1854,color:#fff
        style OBJ fill:#8B5A5A,color:#fff
        style CON fill:#8B5A5A,color:#fff

**Key Properties:**

- Inequality constraints via penalty functions
- Known best solutions from literature
- Mixed continuous design variables
- Realistic engineering specifications
"""


def generate_presets_overview() -> str:
    """Generate overview for Collection suites page."""
    try:
        from surfaces import collection

        n_quick = len(collection.quick)
        n_standard = len(collection.standard)
        n_alg2d = len(collection.algebraic_2d)
        n_algnd = len(collection.algebraic_nd)
        n_bbob = len(collection.bbob)
        n_cec14 = len(collection.cec2014)
        n_cec17 = len(collection.cec2017)
        n_eng = len(collection.engineering)
    except ImportError:
        n_quick = n_standard = n_alg2d = n_algnd = n_bbob = n_cec14 = n_cec17 = n_eng = 0

    return f"""The collection provides curated suites of test function classes for standardized benchmarking.
Using suites ensures comparable results across different studies and projects.

.. mermaid::

    flowchart LR
        CL[collection]

        CL --> QK[quick<br/>{n_quick} functions]
        CL --> ST[standard<br/>{n_standard} functions]
        CL --> A2[algebraic_2d<br/>{n_alg2d} functions]
        CL --> AN[algebraic_nd<br/>{n_algnd} functions]
        CL --> BB[bbob<br/>{n_bbob} functions]
        CL --> C14[cec2014<br/>{n_cec14} functions]
        CL --> C17[cec2017<br/>{n_cec17} functions]
        CL --> EN[engineering<br/>{n_eng} functions]

        QK --> QKd[Fast sanity checks]
        ST --> STd[Diverse landscapes]
        BB --> BBd[COCO benchmark]

        style CL fill:#1A1854,color:#fff
        style QK fill:#5A7A8B,color:#fff
        style ST fill:#5A7A8B,color:#fff
        style A2 fill:#5A7A8B,color:#fff
        style AN fill:#5A7A8B,color:#fff
        style BB fill:#5A7A8B,color:#fff
        style C14 fill:#5A7A8B,color:#fff
        style C17 fill:#5A7A8B,color:#fff
        style EN fill:#5A7A8B,color:#fff

**Usage Pattern:**

Suites contain function **classes**, not instances. Use ``.instantiate()`` to create instances:

.. code-block:: python

    from surfaces import collection

    functions = collection.standard.instantiate(n_dim=10)
    for func in functions:
        result = optimizer.minimize(func)
"""


def generate_noise_overview() -> str:
    """Generate overview for Noise page."""
    return """Noise layers add stochastic disturbances to test function evaluations.
They simulate measurement uncertainty and test algorithm robustness to noisy objectives.

.. mermaid::

    flowchart TD
        N[Noise Layer]

        N --> G[GaussianNoise]
        N --> U[UniformNoise]
        N --> M[MultiplicativeNoise]

        G --> Gf["f(x) + N(0, σ²)"]
        U --> Uf["f(x) + U(low, high)"]
        M --> Mf["f(x) × (1 + N(0, σ²))"]

        subgraph Features
            F1[Configurable intensity]
            F2[Decay schedules]
            F3[Reproducible seeds]
        end

        N --> Features

        style N fill:#1A1854,color:#fff
        style G fill:#5A7A8B,color:#fff
        style U fill:#5A7A8B,color:#fff
        style M fill:#5A7A8B,color:#fff

**Usage:**

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces.modifiers import GaussianNoise

    noise = GaussianNoise(sigma=0.1, seed=42)
    func = SphereFunction(n_dim=2, noise=noise)

    # Evaluations now include noise
    result = func([0.5, 0.5])
"""


def generate_visualization_overview() -> str:
    """Generate overview for Visualization page."""
    return """The visualization module provides plot functions for exploring test function landscapes
and analyzing optimization progress.

.. mermaid::

    flowchart TD
        VIZ[Visualization]

        VIZ --> DISC[Discovery]
        VIZ --> PLOT[Plot Functions]

        DISC --> D1[available_plots]
        DISC --> D2[check_compatibility]
        DISC --> D3[auto_plot]

        PLOT --> P1[plot_surface]
        PLOT --> P2[plot_contour]
        PLOT --> P3[plot_multi_slice]
        PLOT --> P4[plot_convergence]
        PLOT --> P5[plot_fitness_distribution]
        PLOT --> P6[plot_latex]

        P1 --> P1d[3D surface<br/>2D only]
        P2 --> P2d[2D contour<br/>2D only]
        P3 --> P3d[1D slices<br/>Any N-D]
        P6 --> P6d[LaTeX/PDF<br/>With formula]

        style VIZ fill:#1A1854,color:#fff
        style DISC fill:#5A7A8B,color:#fff
        style PLOT fill:#5A7A8B,color:#fff

**Discovery API:**

Use ``available_plots(func)`` to see which plots work with your function:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction
    from surfaces._visualize import available_plots, auto_plot

    func = SphereFunction(n_dim=2)
    print(available_plots(func))  # Lists compatible plots

    fig = auto_plot(func)  # Auto-selects best visualization
    fig.show()
"""


def generate_base_classes_overview() -> str:
    """Generate overview for Base Classes page."""
    counts = count_by_category()

    n_algebraic = (
        counts.get("algebraic_1d", 0)
        + counts.get("algebraic_2d", 0)
        + counts.get("algebraic_nd", 0)
    )
    n_ml = sum(v for k, v in counts.items() if k.startswith("ml_"))
    n_eng = counts.get("engineering", 0)
    n_bbob = counts.get("bbob", 0)
    n_cec = counts.get("cec", 0)

    return f"""Base classes define the interface and shared functionality for all test functions.
Understanding the hierarchy helps when creating custom test functions or extending Surfaces.

.. mermaid::

    classDiagram
        BaseTestFunction <|-- AlgebraicFunction
        BaseTestFunction <|-- MachineLearningFunction
        BaseTestFunction <|-- EngineeringFunction

        AlgebraicFunction <|-- BBOBFunction
        AlgebraicFunction <|-- CECFunction

        CECFunction <|-- CEC2013Function
        CECFunction <|-- CEC2014Function
        CECFunction <|-- CEC2017Function

        MachineLearningFunction <|-- BaseTabular
        MachineLearningFunction <|-- BaseImage
        MachineLearningFunction <|-- BaseTimeSeries

        class BaseTestFunction {{
            +__call__(params)
            +search_space()
            +objective: str
            +noise: BaseNoise
        }}

        class AlgebraicFunction {{
            +latex_formula: str
            +f_global: float
            +x_global: array
        }}

**Function Counts by Base Class:**

- ``AlgebraicFunction``: {n_algebraic} functions
- ``BBOBFunction``: {n_bbob} functions
- ``CECFunction``: {n_cec} functions
- ``MachineLearningFunction``: {n_ml} functions
- ``EngineeringFunction``: {n_eng} functions
"""


def generate_user_guide_overview() -> str:
    """Generate overview for User Guide index page."""
    total = get_total_count()

    return f"""This guide covers practical usage of Surfaces for optimization algorithm development and benchmarking.

Surfaces provides **{total} test functions** with a unified API that works with any optimizer.

.. mermaid::

    flowchart LR
        subgraph Your Code
            OPT[Optimizer]
        end

        subgraph Surfaces
            TF[Test Function]
            TF --> |"__call__"| EVAL[Evaluate]
            TF --> |search_space| SPACE[Get Bounds]
            TF --> |to_scipy| SCIPY[scipy Format]
        end

        OPT --> |params| TF
        EVAL --> |score| OPT

        style OPT fill:#5A8B6A,color:#fff
        style TF fill:#1A1854,color:#fff

**Core Workflow:**

1. **Select** a test function or preset
2. **Configure** dimensions, noise, and callbacks
3. **Integrate** with your optimizer via ``__call__`` or ``to_scipy()``
4. **Visualize** results with the visualization module
"""


def generate_examples_overview() -> str:
    """Generate overview for Examples page."""
    return """These examples demonstrate common usage patterns for Surfaces,
from basic function evaluation to integration with popular optimization libraries.

.. mermaid::

    flowchart TD
        EX[Examples]

        EX --> BASIC[Basic Usage]
        EX --> INT[Integrations]
        EX --> ADV[Advanced]

        BASIC --> B1[Function evaluation]
        BASIC --> B2[Search space access]
        BASIC --> B3[Visualization]

        INT --> I1[Hyperactive]
        INT --> I2[scipy.optimize]
        INT --> I3[Optuna]

        ADV --> A1[Custom functions]
        ADV --> A2[Noise injection]
        ADV --> A3[Benchmarking]

        style EX fill:#1A1854,color:#fff
        style BASIC fill:#5A7A8B,color:#fff
        style INT fill:#5A7A8B,color:#fff
        style ADV fill:#5A7A8B,color:#fff

**Quick Start:**

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Create and evaluate
    func = SphereFunction(n_dim=5)
    score = func([0.1, 0.2, 0.3, 0.4, 0.5])

    # Get search space for optimizer
    bounds = func.search_space()
"""


def main():
    """Generate all diagram/overview files."""
    print(f"Output directory: {DIAGRAMS_DIR}")

    generators = {
        "test_functions_overview.rst": generate_test_functions_overview,
        "algebraic_overview.rst": generate_algebraic_overview,
        "bbob_overview.rst": generate_bbob_overview,
        "cec_overview.rst": generate_cec_overview,
        "ml_overview.rst": generate_ml_overview,
        "engineering_overview.rst": generate_engineering_overview,
        "presets_overview.rst": generate_presets_overview,
        "noise_overview.rst": generate_noise_overview,
        "visualization_overview.rst": generate_visualization_overview,
        "base_classes_overview.rst": generate_base_classes_overview,
        "user_guide_overview.rst": generate_user_guide_overview,
        "examples_overview.rst": generate_examples_overview,
    }

    for filename, generator in generators.items():
        content = generator()
        output_path = DIAGRAMS_DIR / filename
        output_path.write_text(content)
        print(f"  Generated: {filename}")


if __name__ == "__main__":
    main()
