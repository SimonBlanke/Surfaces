.. _minimal_dependencies:

====================
Minimal Dependencies
====================

Surfaces is designed to be lightweight. The core installation requires
only **numpy** as a dependency. Everything else is optional.

----

Core Installation
=================

.. code-block:: bash

    pip install surfaces

This installs:

- **numpy**: The only required dependency
- **Algebraic functions**: All mathematical test functions (1D, 2D, N-D)
- **Engineering functions**: All constrained design problems
- **BBOB functions**: Black-Box Optimization Benchmarking suite
- **CEC functions**: Competition on Evolutionary Computation benchmarks

With just numpy, you get access to the majority of test functions
in the library.

----

Why Minimal Matters
===================

CI/CD Pipelines
---------------

Lightweight dependencies mean faster CI/CD pipelines:

.. code-block:: yaml

    # GitHub Actions example
    - name: Install test dependencies
      run: pip install surfaces  # Fast, minimal install

    - name: Run optimizer benchmarks
      run: python benchmark.py

No waiting for heavy dependencies like scikit-learn or TensorFlow
to install when you only need algebraic functions.

Reproducibility
---------------

Fewer dependencies mean fewer version conflicts:

.. code-block:: bash

    # Only one dependency to pin
    surfaces==0.6.0
    numpy>=1.18.0

Clean Environments
------------------

Keep your virtual environments small:

.. code-block:: bash

    $ pip install surfaces
    $ pip list
    Package    Version
    ---------- -------
    numpy      1.24.0
    surfaces   0.6.0

----

Optional Features
=================

Add features only when you need them:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Extra
     - Command
     - What You Get
   * - **ml**
     - ``pip install surfaces[ml]``
     - Machine Learning test functions (requires scikit-learn)
   * - **surrogates**
     - ``pip install surfaces[surrogates]``
     - Pre-trained surrogate models (requires onnxruntime)
   * - **viz**
     - ``pip install surfaces[viz]``
     - Visualization tools (requires plotly, matplotlib)
   * - **full**
     - ``pip install surfaces[full]``
     - All optional features

----

Installation Guide
==================

For detailed installation instructions for each optional feature:

.. grid:: 2 2 2 2
   :gutter: 3

   .. grid-item-card:: CEC Functions
      :link: /installation/cec
      :link-type: doc

      Additional dependencies for CEC benchmark suites.

   .. grid-item-card:: Machine Learning
      :link: /installation/machine_learning
      :link-type: doc

      scikit-learn, XGBoost, and ML-based functions.

   .. grid-item-card:: Surrogates
      :link: /installation/surrogates
      :link-type: doc

      ONNX runtime and pre-trained models.

   .. grid-item-card:: Visualization
      :link: /installation/visualization
      :link-type: doc

      Plotly and matplotlib for surface plots.

----

Quick Check
===========

Verify your installation:

.. code-block:: python

    import surfaces
    print(f"Surfaces version: {surfaces.__version__}")

    # Core functions work with minimal install
    from surfaces.test_functions import SphereFunction
    func = SphereFunction(n_dim=3)
    print(f"Sphere(1,1,1) = {func({'x0': 1, 'x1': 1, 'x2': 1})}")

----

Next Steps
==========

- :doc:`/installation` - Complete installation guide
- :doc:`/get_started` - Get started with Surfaces
