.. _user_guide_benchmark:

============
Benchmarking
============

The ``surfaces.benchmark`` module provides a structured way to compare
optimization algorithms. It handles the execution loop, cost measurement
in Compute Units, result storage, statistical analysis, and visualization.

You define which functions to test and which optimizers to compare. The
module runs all combinations, records every evaluation, and gives you
tools to analyze the results: summary tables, Expected Running Time,
statistical rankings with multiple-comparison correction, and
publication-style plots including Critical Difference diagrams.


Quick Start
===========

.. code-block:: python

    from surfaces.benchmark import Benchmark
    from surfaces import collection

    bench = Benchmark(budget_cu=50_000, n_seeds=5)
    bench.add_functions(collection.filter(category="bbob", n_dim=2))
    bench.add_optimizers([HillClimbingOptimizer, RandomSearchOptimizer])
    bench.run()

    print(bench.results.summary())

The ``Benchmark`` object is the central entry point. It stores configuration,
registered functions and optimizers, and all accumulated traces. Calling
``run()`` only executes combinations that have no trace yet, so you can
add more optimizers later and re-run without repeating finished work.


Adding Functions
================

Functions can be added as classes, lists of classes, or filtered from
the collection:

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction, AckleyFunction

    bench.add_functions([SphereFunction, AckleyFunction])

    # Or use the collection system
    from surfaces import collection
    bench.add_functions(collection.filter(category="bbob"))
    bench.add_functions(collection.filter(eval_cost=lambda c: c is not None and c < 10))

Functions are instantiated with default parameters for each trial. The
benchmark uses the class (not an instance) so it can create fresh
instances per seed.


Adding Optimizers
=================

Optimizers are auto-detected by module path. The module ships adapters
for Gradient-Free-Optimizers, Optuna, scipy, CMA-ES, Nevergrad, SMAC,
scikit-optimize, BayesOpt, pymoo, and PySwarms. Any optimizer with an
``ask()``/``tell()`` interface works automatically via the generic adapter.

.. code-block:: python

    # Bare class (default parameters)
    bench.add_optimizers(HillClimbingOptimizer)

    # With parameters
    bench.add_optimizers((TPESampler, {"n_startup_trials": 10}))

    # Multiple at once
    bench.add_optimizers([
        HillClimbingOptimizer,
        RandomSearchOptimizer,
        (BayesianOptimizer, {"xi": 0.01}),
    ])


Running Benchmarks
==================

.. code-block:: python

    bench.run()

Each call to ``run()`` executes only the missing ``(function, optimizer, seed)``
combinations. This makes incremental benchmarking natural: add a new optimizer,
call ``run()`` again, and only the new combinations are executed.

Budget Control
--------------

You can limit runs by Compute Units, iteration count, or both:

.. code-block:: python

    # Stop after 50,000 CU (includes optimizer overhead)
    bench = Benchmark(budget_cu=50_000, n_seeds=5)

    # Stop after 1000 evaluations
    bench = Benchmark(budget_iter=1000, n_seeds=5)

    # Whichever limit is reached first
    bench = Benchmark(budget_cu=50_000, budget_iter=1000, n_seeds=5)

When using CU budgets, the benchmark tracks both function evaluation cost
and optimizer overhead, giving a hardware-independent measure of total
computational effort. See :ref:`user_guide_compute_units` for details on
how Compute Units work.

Error Handling
--------------

The ``catch`` parameter controls what happens when a trial fails:

.. code-block:: python

    bench = Benchmark(budget_cu=50_000, catch="warn")

``"raise"`` (default) stops immediately on the first error.
``"warn"`` logs a warning and continues with other trials.
``"skip"`` silently skips the failed trial.

Failed trials are always recorded in ``bench.errors`` regardless of mode.


Analyzing Results
=================

All analysis methods live on the ``bench.results`` accessor.

Summary Table
-------------

.. code-block:: python

    print(bench.results.summary())

Prints a formatted table with mean best score, evaluation count, and
overhead percentage for each (function, optimizer) pair. Use ``show_ci=True``
to include standard deviations and 95% confidence intervals across seeds.
``at_cu`` or ``at_iter`` report scores at a specific budget point instead of
the final result.

Expected Running Time (ERT)
---------------------------

ERT follows the COCO convention: total budget across all seeds divided by
the number of successful seeds (those that reached the target). A problem
counts as "solved" when ``best_so_far <= f_global + precision``.

.. code-block:: python

    ert = bench.results.ert(precision=1.0)
    print(ert)

    # Subscript access
    entry = ert["AckleyFunction"]["HillClimbing"]
    print(entry.ert_cu, entry.solved, entry.total)

    # Export to pandas
    df = ert.to_dataframe()

The ``ERTTable`` is printable, subscriptable by function and optimizer name,
and exportable to a DataFrame.

Optimizer Ranking
-----------------

Ranks optimizers by normalized performance across all functions. Scores are
normalized per function (0 = worst observed, 1 = best observed) and averaged
over seeds. Pairwise Wilcoxon signed-rank tests assess whether differences
are statistically significant.

.. code-block:: python

    ranking = bench.results.ranking()
    print(ranking)

By default, p-values are corrected using the Holm step-down procedure to
control the family-wise error rate across multiple comparisons. Pass
``correction=None`` for raw uncorrected p-values.

.. code-block:: python

    # With Holm correction (default)
    ranking = bench.results.ranking(correction="holm")

    # Raw p-values (not recommended for more than 2 optimizers)
    ranking = bench.results.ranking(correction=None)


Statistical Comparison
======================

When comparing three or more optimizers, the recommended workflow follows
Demsar (2006):

1. Run the **Friedman omnibus test** to check whether any difference exists.
2. If significant, run **pairwise post-hoc tests** with correction.
3. Visualize with a **Critical Difference diagram**.

Friedman Test
-------------

The Friedman test is a non-parametric test that checks whether at least one
optimizer differs significantly from the others. It operates on the
per-function rank matrix.

.. code-block:: python

    friedman = bench.results.friedman()
    print(friedman)

The output includes both the standard chi-squared statistic and the
Iman-Davenport F-statistic, which is less conservative (more powerful).
The ``significant`` property tells you whether to proceed with post-hoc
tests.

.. code-block:: python

    if friedman.significant:
        ranking = bench.results.ranking()  # Holm-corrected p-values
        print(ranking)
    else:
        print("No significant differences found.")

The Friedman test requires at least 3 optimizers and at least 3 functions
where all optimizers produced results (complete blocks). Average ranks
are computed with proper tied-rank handling.

Holm Correction
---------------

Running multiple pairwise comparisons inflates the chance of false positives.
With 5 optimizers you have 10 pairwise tests; at alpha=0.05, the probability
of at least one false positive rises to roughly 40%.

The Holm step-down correction adjusts p-values upward to compensate. It is
strictly more powerful than Bonferroni correction while maintaining the same
family-wise error rate guarantee. The correction is applied by default in
``bench.results.ranking()``.


Visualization
=============

All visualization methods live on the ``bench.plot`` accessor and require
the ``viz`` extra (``pip install surfaces[viz]``).

ECDF Plot
---------

The Empirical Cumulative Distribution Function shows what fraction of
problems each optimizer solved within a given CU budget. A problem counts
as "solved" when the best score reaches ``f_global + precision``.

.. code-block:: python

    fig = bench.plot.ecdf(precision=1.0)
    fig.show()

    # Multiple precision levels as stacked subplots
    fig = bench.plot.ecdf(precision=[1.0, 0.1, 0.01])
    fig.show()

ECDF plots use Plotly and return a ``plotly.graph_objects.Figure``.

Convergence Plot
----------------

Shows how the best-found score evolves over the CU budget for a single
function, aggregated across seeds.

.. code-block:: python

    fig = bench.plot.convergence("AckleyFunction")
    fig.show()

    # Customize center line and uncertainty band
    fig = bench.plot.convergence(
        "AckleyFunction",
        center="median",  # or "mean"
        band="iqr",       # or "minmax", "std", None
        log_y=True,
    )

The default shows the median with an interquartile range band, which is
robust to outliers.

Critical Difference Diagram
---------------------------

The CD diagram (Demsar, 2006) visualizes optimizer rankings on a horizontal
axis. Algorithms that are not statistically distinguishable are connected
by thick bars.

.. code-block:: python

    fig = bench.plot.cd_diagram()
    fig.savefig("cd_diagram.pdf")

This computes Friedman-style average ranks (with tied-rank handling on
complete blocks) and uses Holm-corrected Wilcoxon p-values to determine
which groups of optimizers are statistically indistinguishable. The result
is a matplotlib Figure suitable for publication.

.. code-block:: python

    # Customize
    fig = bench.plot.cd_diagram(
        alpha=0.05,
        correction="holm",
        title="BBOB Suite Comparison",
        width=10.0,
    )


Persistence
===========

Save the full benchmark state (configuration, function/optimizer registry,
and all traces) to a JSON file:

.. code-block:: python

    bench.io.save("results.json")

    # Later, restore everything
    bench = Benchmark.load("results.json")
    print(bench.results.summary())

Loading checks the Surfaces version and emits a warning if it differs from
the version used when saving. All traces are reconstructed, so you can
continue analysis or add more optimizers and re-run.


Pre-defined Suites
==================

Suites are named configurations with pre-selected function filters and
budget defaults:

.. code-block:: python

    from surfaces.benchmark import suites

    bench = Benchmark.from_suite(suites["bbob_standard"])
    bench.add_optimizers([...])
    bench.run()

Available suites:

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Suite
     - Budget
     - Description
   * - ``quick``
     - 10,000 CU
     - Small set of fast algebraic functions for rapid testing.
   * - ``bbob_standard``
     - 100,000 CU
     - Standard 2D BBOB suite (24 functions).
   * - ``overhead_analysis``
     - 5,000 CU
     - Very cheap functions where optimizer overhead dominates.
   * - ``expensive_functions``
     - 500,000 CU
     - Costly functions where smart optimization matters most.

Suite defaults can be overridden:

.. code-block:: python

    bench = Benchmark.from_suite(suites["bbob_standard"], n_seeds=10, budget_cu=200_000)


Parallel Execution
==================

For large benchmarks, pass a backend to ``run()`` to distribute trials
across processes or threads:

.. code-block:: python

    from surfaces.benchmark import ProcessBackend

    bench.run(backend=ProcessBackend(n_jobs=4))

``ProcessBackend`` spawns separate processes (bypasses the GIL, requires
picklable optimizers). ``ThreadBackend`` uses threads (useful when the
bottleneck releases the GIL, such as C-extension-based optimizers).
Both accept ``n_jobs=-1`` to use all available cores.

.. warning::

   Parallel execution collects results after all trials complete. With
   ``catch="raise"``, the first error is raised only after the entire
   batch finishes, not immediately.
