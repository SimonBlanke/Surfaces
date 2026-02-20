.. _home:

.. raw:: html

   <div class="hero-section">
      <div class="hero-content">
         <h1 class="hero-title">Surfaces</h1>
         <p class="hero-tagline">Black-box optimization test functions for benchmarking</p>
      </div>
   </div>

   <div class="stats-strip">
      <a href="get_started/curated_test_functions.html" class="stat-item">
         <div class="stat-front">
            <div class="stat-value">Curated</div>
            <div class="stat-label">Test Functions</div>
         </div>
         <div class="stat-hover">
            <div class="stat-hover-text">Algebraic, ML, Engineering, CEC and BBOB benchmarks</div>
         </div>
      </a>
      <a href="get_started/plug_and_play_integration.html" class="stat-item">
         <div class="stat-front">
            <div class="stat-value">Plug & Play</div>
            <div class="stat-label">Integration</div>
         </div>
         <div class="stat-hover">
            <div class="stat-hover-text">Works with scipy, Optuna, Ray Tune, SMAC and more</div>
         </div>
      </a>
      <a href="get_started/machine_learning_accelerated.html" class="stat-item">
         <div class="stat-front">
            <div class="stat-value">ML-Model</div>
            <div class="stat-label">Accelerated</div>
         </div>
         <div class="stat-hover">
            <div class="stat-hover-text">ONNX surrogates for fast function evaluation</div>
         </div>
      </a>
      <a href="get_started/minimal_dependencies.html" class="stat-item">
         <div class="stat-front">
            <div class="stat-value">Minimal</div>
            <div class="stat-label">Dependencies</div>
         </div>
         <div class="stat-hover">
            <div class="stat-hover-text">Core library requires only numpy</div>
         </div>
      </a>
   </div>

   <p align="center">
   <a href="https://github.com/SimonBlanke/Surfaces/actions"><img src="https://img.shields.io/github/actions/workflow/status/SimonBlanke/Surfaces/tests.yml?style=for-the-badge&logo=githubactions&logoColor=white&label=tests" alt="Tests"></a>
   <a href="https://app.codecov.io/gh/SimonBlanke/Surfaces"><img src="https://img.shields.io/codecov/c/github/SimonBlanke/Surfaces?style=for-the-badge&logo=codecov&logoColor=white" alt="Coverage"></a>
   </p>

   <hr style="border-radius: 0; border-top: 3px solid var(--pst-color-border); margin: 2.5rem 0 0.3rem 0;">

   <p style="text-align: center; max-width: 800px; margin: 0.5rem auto; line-height: 1.6;">
   Surfaces provides a curated collection of optimization test functions for benchmarking
   gradient-free and black-box optimizers. It includes classical mathematical benchmarks,
   N-dimensional scalable functions, and ML-model-accelerated surrogates, all accessible
   through a minimal, plug-and-play API.
   </p>

   <div style="margin-bottom: 8rem;"></div>

.. _features:

Features
========

What makes Surfaces ideal for optimization benchmarking.

.. grid:: 1 2 3 3
   :gutter: 4

   .. grid-item-card::
      :class-card: feature-card

      **|n_total_functions|+ Test Functions**
      ^^^
      Classic optimization functions from the literature,
      ML-based functions, and engineering benchmarks.

      +++
      :doc:`Explore all function categories <user_guide/introduction>`

   .. grid-item-card::
      :class-card: feature-card

      **ML-Based Functions**
      ^^^
      Test functions based on real machine learning models:
      hyperparameter tuning as optimization benchmark.

      +++
      :doc:`K-Neighbors, Gradient Boosting <user_guide/machine_learning>`

   .. grid-item-card::
      :class-card: feature-card

      **scipy Integration**
      ^^^
      Convert any function to scipy.optimize format with
      ``to_scipy()`` for seamless integration.

      +++
      :doc:`Works with scipy optimizers <user_guide/scipy_integration>`

   .. grid-item-card::
      :class-card: feature-card

      **Flexible Evaluation**
      ^^^
      Call functions with dictionaries, keyword arguments,
      numpy arrays, or batch evaluation.

      +++
      :doc:`Multiple evaluation interfaces <user_guide/introduction>`

   .. grid-item-card::
      :class-card: feature-card

      **Loss or Score**
      ^^^
      Every function supports both minimization (loss)
      and maximization (score) modes.

      +++
      :doc:`Metric handling <user_guide/introduction>`

   .. grid-item-card::
      :class-card: feature-card

      **Built-in Visualization**
      ^^^
      Plotly-based surface plots and heatmaps
      for 2D function visualization.

      +++
      :doc:`Visualization tools <user_guide/visualization>`

----

Function Categories
===================

Surfaces provides functions in three main categories, each with a consistent interface.

.. grid:: 1 1 3 3
   :gutter: 4

   .. grid-item-card::
      :class-card: category-card category-card-math

      Algebraic Functions (|n_algebraic|)
      ^^^
      Classic test functions from the optimization literature with
      known global optima and analytical formulas.

      - **1D**: |n_1d| functions
      - **2D**: |n_2d| functions
      - **N-D**: |n_nd| scalable functions

      +++
      :doc:`Explore algebraic functions <user_guide/mathematical>`

   .. grid-item-card::
      :class-card: category-card category-card-ml

      Machine Learning Functions (|n_ml|)
      ^^^
      Test functions based on real ML model training tasks,
      providing realistic hyperparameter optimization landscapes.

      - **Classification**: Tabular, Image, Time Series
      - **Regression**: Tabular models
      - **Forecasting**: Time series

      +++
      :doc:`Explore ML functions <user_guide/machine_learning>`

   .. grid-item-card::
      :class-card: category-card category-card-eng

      Engineering Functions (|n_engineering|)
      ^^^
      Real-world constrained engineering design optimization
      problems with physical meaning.

      - Welded Beam, Pressure Vessel
      - Tension-Compression Spring
      - Cantilever Beam, Three-Bar Truss

      +++
      :doc:`Explore engineering functions <user_guide/engineering>`

----

Quick Install
=============

.. raw:: html

   <p align="center">
   <a href="https://pypi.org/project/surfaces/" target="_blank"><img src="https://img.shields.io/pypi/v/surfaces?style=flat-square" alt="PyPI Version"></a>
   <a href="https://pypi.org/project/surfaces/" target="_blank"><img src="https://img.shields.io/pypi/pyversions/surfaces?style=flat-square" alt="Python Versions"></a>
   <a href="https://github.com/SimonBlanke/Surfaces/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/github/license/SimonBlanke/Surfaces?style=flat-square" alt="License"></a>
   </p>

   <div class="segmented-tabs" id="install-tabs">
      <nav class="segmented-tabs-nav" role="tablist">
         <button class="segmented-tab-btn active" role="tab" aria-selected="true" aria-controls="install-panel-pip">
            pip install
         </button>
         <button class="segmented-tab-btn" role="tab" aria-selected="false" aria-controls="install-panel-dev">
            Development
         </button>
      </nav>
      <div class="segmented-tabs-content">
         <div class="segmented-tab-panel active" id="install-panel-pip" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install surfaces</pre></div>
         </div>
         <div class="segmented-tab-panel" id="install-panel-dev" role="tabpanel">
            <div class="highlight"><pre><span class="gp">$ </span>pip install surfaces[dev,test]</pre></div>
         </div>
      </div>
   </div>

----

Quick Example
=============

Get started in just a few lines of code:

.. raw:: html

   <div class="vertical-tabs" id="example-tabs">
      <nav class="vertical-tabs-nav" role="tablist">
         <button class="vertical-tab-btn active" role="tab" aria-selected="true" aria-controls="example-panel-basic" data-tab="example-basic">
            <span class="tab-indicator"></span>
            <span>Basic Usage</span>
         </button>
         <button class="vertical-tab-btn" role="tab" aria-selected="false" aria-controls="example-panel-scipy" data-tab="example-scipy">
            <span class="tab-indicator"></span>
            <span>scipy Integration</span>
         </button>
         <button class="vertical-tab-btn" role="tab" aria-selected="false" aria-controls="example-panel-ml" data-tab="example-ml">
            <span class="tab-indicator"></span>
            <span>ML Functions</span>
         </button>
      </nav>
      <div class="vertical-tabs-content">
         <div class="vertical-tab-panel active" id="example-panel-basic" role="tabpanel">

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    # Create a 3-dimensional Sphere function
    func = SphereFunction(n_dim=3)

    # Evaluate with a dictionary
    result = func({"x0": 1.0, "x1": 2.0, "x2": 3.0})
    print(f"Loss: {result}")  # Loss: 14.0

    # Get the default search space
    search_space = func.search_space()
    # {"x0": array(...), "x1": array(...), "x2": array(...)}

.. raw:: html

         </div>
         <div class="vertical-tab-panel" id="example-panel-scipy" role="tabpanel">

.. code-block:: python

    from surfaces.test_functions.algebraic import RosenbrockFunction
    from scipy.optimize import minimize

    # Create the test function
    func = RosenbrockFunction(n_dim=5)

    # Convert to scipy format
    objective, bounds, x0 = func.to_scipy()

    # Run scipy optimizer
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    print(f"Optimal x: {result.x}")
    print(f"Minimum value: {result.fun}")

.. raw:: html

         </div>
         <div class="vertical-tab-panel" id="example-panel-ml" role="tabpanel">

.. code-block:: python

    from surfaces.test_functions.machine_learning import KNeighborsClassifierFunction

    # Create an ML-based test function
    func = KNeighborsClassifierFunction()

    # Evaluate with hyperparameters
    score = func({
        "n_neighbors": 5,
        "weights": "distance",
        "p": 2
    })
    print(f"Accuracy: {score}")

    # Get the hyperparameter search space
    search_space = func.search_space()

.. raw:: html

         </div>
      </div>
   </div>

   <script>
   document.addEventListener('DOMContentLoaded', function() {
      // Handle vertical tabs (Quick Example)
      document.querySelectorAll('.vertical-tabs').forEach(function(tabContainer) {
         const buttons = tabContainer.querySelectorAll('.vertical-tab-btn');
         const panels = tabContainer.querySelectorAll('.vertical-tab-panel');

         buttons.forEach(function(button) {
            button.addEventListener('click', function() {
               buttons.forEach(btn => {
                  btn.classList.remove('active');
                  btn.setAttribute('aria-selected', 'false');
               });
               panels.forEach(panel => panel.classList.remove('active'));

               this.classList.add('active');
               this.setAttribute('aria-selected', 'true');

               const panelId = this.getAttribute('aria-controls');
               const panel = document.getElementById(panelId);
               if (panel) {
                  panel.classList.add('active');
               }
            });
         });
      });

      // Handle segmented tabs (Quick Install)
      document.querySelectorAll('.segmented-tabs').forEach(function(tabContainer) {
         const buttons = tabContainer.querySelectorAll('.segmented-tab-btn');
         const panels = tabContainer.querySelectorAll('.segmented-tab-panel');

         buttons.forEach(function(button) {
            button.addEventListener('click', function() {
               buttons.forEach(btn => {
                  btn.classList.remove('active');
                  btn.setAttribute('aria-selected', 'false');
               });
               panels.forEach(panel => panel.classList.remove('active'));

               this.classList.add('active');
               this.setAttribute('aria-selected', 'true');

               const panelId = this.getAttribute('aria-controls');
               const panel = document.getElementById(panelId);
               if (panel) {
                  panel.classList.add('active');
               }
            });
         });
      });

      // Back to Top - inject into sidebar
      const sidebar = document.querySelector('.bd-toc-nav.page-toc');
      if (sidebar) {
         const backToTopDiv = document.createElement('div');
         backToTopDiv.className = 'back-to-top-sidebar';
         backToTopDiv.innerHTML = '<a href="#">Back to top</a>';
         sidebar.appendChild(backToTopDiv);

         backToTopDiv.querySelector('a').addEventListener('click', function(e) {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
         });

         window.addEventListener('scroll', function() {
            if (window.scrollY > 400) {
               backToTopDiv.classList.add('visible');
            } else {
               backToTopDiv.classList.remove('visible');
            }
         });
      }
   });
   </script>

----

Contents
========

.. toctree::
   :maxdepth: 2
   :hidden:

   get_started
   installation
   user_guide
   api_reference
   examples
   faq
   troubleshooting
   get_involved
   about

.. raw:: html

   <div class="contents-grid">
      <a href="get_started.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Get Started</div>
               <div class="contents-card-desc">Quick introduction to Surfaces</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
      <a href="installation.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Installation</div>
               <div class="contents-card-desc">Setup guide and requirements</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
      <a href="user_guide.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">User Guide</div>
               <div class="contents-card-desc">In-depth tutorials and explanations</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
      <a href="api_reference.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">API Reference</div>
               <div class="contents-card-desc">Technical reference for all classes</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
      <a href="examples.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Examples</div>
               <div class="contents-card-desc">Code examples and use cases</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
      <a href="get_involved.html" class="contents-card">
         <div class="contents-card-inner">
            <div class="contents-card-text">
               <div class="contents-card-title">Get Involved</div>
               <div class="contents-card-desc">Contribute to Surfaces</div>
            </div>
            <div class="contents-card-arrow">...</div>
         </div>
      </a>
   </div>
