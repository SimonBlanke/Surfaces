.. _user_guide_multi_fidelity:

===============
Multi-Fidelity
===============

Multi-fidelity optimization algorithms like Hyperband, BOHB and ASHA
evaluate configurations at varying levels of accuracy. A cheap low-fidelity
evaluation (e.g. training on 10% of the data) filters out bad candidates
early, while full-fidelity evaluations are reserved for promising ones.

Surfaces supports this pattern through the ``fidelity`` parameter on all
ML test functions.


Basic Usage
===========

Pass ``fidelity`` as a keyword argument to any ML function call. The value
must be in the range (0, 1], where 1.0 means full evaluation and smaller
values use proportionally less training data.

.. code-block:: python

    from surfaces.test_functions import RandomForestClassifierFunction

    func = RandomForestClassifierFunction(dataset="digits", cv=5)

    # Full-fidelity evaluation (default behaviour, same as fidelity=1.0)
    score_full = func({"n_estimators": 100, "max_depth": 10})

    # Low-fidelity: train on 10% of the data
    score_cheap = func({"n_estimators": 100, "max_depth": 10}, fidelity=0.1)

    # Medium-fidelity: train on 50% of the data
    score_mid = func({"n_estimators": 100, "max_depth": 10}, fidelity=0.5)

Calling without ``fidelity`` or with ``fidelity=None`` gives the same
result as before this feature existed. Existing code is not affected.


Subsampling Strategies
======================

Different ML function categories use different subsampling strategies to
ensure that the reduced dataset remains meaningful.

.. list-table::
   :widths: 30 20 50
   :header-rows: 1

   * - Category
     - Strategy
     - Rationale
   * - Tabular Classification
     - Stratified
     - Preserves class distribution (important for imbalanced or sorted datasets like iris)
   * - Tabular Regression
     - Shuffled
     - Deterministic random subset with fixed seed
   * - Time-Series Forecasting
     - Sequential
     - Takes the first N% of the series to preserve temporal order
   * - Time-Series Classification
     - Stratified
     - Same as tabular classification
   * - Image Classification
     - Stratified
     - Same as tabular classification
   * - Ensemble Optimization
     - Stratified
     - Uses classification datasets internally
   * - Feature Engineering
     - Stratified
     - Uses classification datasets internally

All subsampling uses ``random_state=42`` for full reproducibility. The same
fidelity value always produces the same subset.


Successive Halving Example
==========================

A typical Successive Halving pattern evaluates many configurations cheaply,
then narrows down:

.. code-block:: python

    import numpy as np
    from surfaces.test_functions import GradientBoostingClassifierFunction

    func = GradientBoostingClassifierFunction(dataset="digits", cv=3)

    # Generate random configurations
    rng = np.random.RandomState(0)
    configs = [
        {"n_estimators": int(rng.choice(func.search_space["n_estimators"])),
         "max_depth": int(rng.choice(func.search_space["max_depth"])),
         "learning_rate": rng.choice(func.search_space["learning_rate"])}
        for _ in range(27)
    ]

    # Round 1: evaluate all 27 at low fidelity
    scores = [(c, func(c, fidelity=0.1)) for c in configs]
    top_9 = sorted(scores, key=lambda x: x[1])[:9]

    # Round 2: evaluate top 9 at medium fidelity
    scores = [(c, func(c, fidelity=0.3)) for c in [t[0] for t in top_9]]
    top_3 = sorted(scores, key=lambda x: x[1])[:3]

    # Round 3: evaluate top 3 at full fidelity
    scores = [(c, func(c, fidelity=1.0)) for c in [t[0] for t in top_3]]
    best = min(scores, key=lambda x: x[1])


Memory Cache
============

When ``memory=True``, the cache distinguishes between fidelity levels.
The same hyperparameters evaluated at different fidelities produce
separate cache entries:

.. code-block:: python

    func = GradientBoostingClassifierFunction(dataset="digits", memory=True)

    params = {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}

    func(params, fidelity=0.1)   # computed
    func(params, fidelity=0.1)   # cache hit
    func(params, fidelity=1.0)   # computed (different fidelity)
    func(params)                 # computed (fidelity=None, separate key)


Data Collection
===============

When ``fidelity`` is set, the recorded evaluation data includes the
fidelity value:

.. code-block:: python

    func = GradientBoostingClassifierFunction(dataset="digits")
    func({"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1}, fidelity=0.3)

    print(func.data.search_data[-1])
    # {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1,
    #  'score': -0.87, 'fidelity': 0.3}

Evaluations without fidelity do not include the key in the record,
keeping backwards compatibility with existing data processing code.


Limitations
===========

**Surrogates do not support fidelity.** When ``use_surrogate=True``, the
surrogate always returns full-fidelity predictions regardless of the
fidelity value. A ``UserWarning`` is raised in this case.

**Very low fidelity on small datasets may fail.** The subsampled data must
have at least as many samples as the number of CV folds. For example, iris
(150 samples, 3 classes) with ``cv=5`` and ``fidelity=0.05`` produces only
about 8 samples, which is not enough for 5-fold cross-validation. A
``ValueError`` with a clear message is raised when this happens.

**Algebraic functions ignore fidelity.** Passing ``fidelity`` to an
algebraic function like ``SphereFunction`` has no effect. The parameter
is accepted without error (for API uniformity) but does not change the
evaluation.
