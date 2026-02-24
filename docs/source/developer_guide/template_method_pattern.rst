.. _developer_template_method:

======================
Template Method Pattern
======================

Surfaces uses the **template method pattern** throughout its class hierarchy.
Public methods are fixed in base classes and delegate to private override
points that subclasses implement. This separation keeps the public API stable
while allowing each function category to customize its internals.

Design Principle
================

Every public method that subclasses need to customize follows this structure:

.. code-block:: text

    BaseClass
        public_method()        <-- fixed, handles validation/orchestration
            calls _hook()      <-- override point, subclasses implement this

Users call the public method. Contributors implement the private hook.

Overview
========

.. list-table::
   :header-rows: 1
   :widths: 30 30 20

   * - Public API (fixed)
     - Override Point (implement)
     - Defined In
   * - ``__call__(params)``
     - ``_objective(params)``
     - ``BaseTestFunction``
   * - ``pure(params)``
     - ``_objective(params)``
     - ``BaseTestFunction``
   * - ``batch(X)``
     - ``_batch_objective(X)``
     - ``BaseTestFunction``
   * - ``search_space`` (property)
     - ``_default_search_space()``
     - ``BaseTestFunction``
   * - ``raw_objective(params)``
     - ``_raw_objective(params)``
     - ``EngineeringFunction``
   * - ``constraints(params)``
     - ``_constraints(params)``
     - ``EngineeringFunction``

----

BaseTestFunction
================

The root class defines the core template methods.

``__call__`` and ``_objective``
-------------------------------

``__call__`` handles input normalization, memory caching, modifier application,
objective direction (minimize/maximize), data collection, and callbacks.
Subclasses only implement ``_objective(params) -> float``.

.. code-block:: python

    class BaseTestFunction:
        def __call__(self, params):
            params = self._normalize_input(params)
            # ... memory check, timing, error handling ...
            result = self._objective(params)    # <-- override point
            # ... modifiers, direction flip, recording ...
            return result

        def _objective(self, params):
            raise NotImplementedError(
                f"{type(self).__name__} must implement _objective(self, params)"
            )

``batch`` and ``_batch_objective``
----------------------------------

``batch`` handles input validation (shape, dimensions) and objective direction.
Subclasses implement ``_batch_objective(X) -> ArrayLike`` for vectorized
evaluation.

.. code-block:: python

    class BaseTestFunction:
        def _batch_objective(self, X):
            raise NotImplementedError(
                f"{type(self).__name__} does not support vectorized batch evaluation. "
                "Implement _batch_objective(X) to enable this feature."
            )

        def batch(self, X):
            # ... validation (shape, n_dim) ...
            result = self._batch_objective(X)   # <-- override point
            if self.objective == "maximize":
                result = -result
            return result

Not all functions support batch evaluation. Functions that do (algebraic, BBOB,
CEC, engineering) implement ``_batch_objective`` with vectorized numpy
operations. Functions that don't (ML) raise ``NotImplementedError``.

``search_space`` and ``_default_search_space``
----------------------------------------------

``search_space`` is a read-only property. Subclasses define
``_default_search_space()`` to return the parameter space dict.

.. code-block:: python

    class BaseTestFunction:
        def _default_search_space(self):
            raise NotImplementedError(
                "'_default_search_space' must be implemented"
            )

        @property
        def search_space(self):
            return self._default_search_space()

Each base class builds the search space differently:

- **AlgebraicFunction**: from ``default_bounds`` and ``n_dim``
- **EngineeringFunction**: from ``variable_names`` and ``variable_bounds``
- **MachineLearningFunction**: from ``para_names`` and ``*_default`` class attributes
- **SimulationFunction**: defined per function (explicit dict)

----

EngineeringFunction
===================

Engineering functions add a second template layer on top of ``BaseTestFunction``.
The ``_objective`` method is implemented as a sub-template that combines the raw
objective with a constraint penalty.

``raw_objective`` and ``_raw_objective``
----------------------------------------

.. code-block:: python

    class EngineeringFunction(BaseTestFunction):
        def _raw_objective(self, params):
            raise NotImplementedError(
                f"{type(self).__name__} must implement _raw_objective(self, params)"
            )

        def raw_objective(self, params):
            """Public API: evaluate raw objective without penalties."""
            return self._raw_objective(params)

        def _objective(self, params):
            """Sub-template: raw objective + penalty."""
            return self._raw_objective(params) + self.penalty(params)

``constraints`` and ``_constraints``
------------------------------------

.. code-block:: python

    class EngineeringFunction(BaseTestFunction):
        def _constraints(self, params):
            """Override in subclasses. Default: no constraints."""
            return []

        def constraints(self, params):
            """Public API: evaluate constraint functions."""
            return self._constraints(params)

        def constraint_violations(self, params):
            return [max(0, g) for g in self._constraints(params)]

        def is_feasible(self, params):
            return all(g <= 0 for g in self._constraints(params))

The internal methods (``constraint_violations``, ``is_feasible``, ``penalty``)
call ``_constraints`` directly, bypassing the public wrapper. This avoids
redundant indirection within the framework.

----

MachineLearningFunction
=======================

ML functions use a similar sub-template. The override point is
``_ml_objective(params)``.

.. code-block:: python

    class MachineLearningFunction(BaseTestFunction):
        def _ml_objective(self, params):
            raise NotImplementedError(
                f"{type(self).__name__} must implement _ml_objective(self, params)"
            )

        def _objective(self, params):
            return self._ml_objective(params)

ML functions also override ``_evaluate`` to handle surrogate models and the
inverted minimize/maximize convention (ML scores are "higher is better").

----

SimulationFunction
==================

Simulation functions split ``_objective`` into a three-step pipeline:

.. code-block:: python

    class SimulationFunction(BaseTestFunction):
        def _objective(self, params):
            result = self._run_simulation(params)
            return self._extract_objective(result)

Subclasses implement:

1. ``_setup_simulation()`` -- called once during ``__init__``
2. ``_run_simulation(params)`` -- execute the simulation
3. ``_extract_objective(result)`` -- extract scalar from simulation output

The ``ODESimulationFunction`` further specializes this by providing
``_run_simulation`` (scipy ODE integration) and requiring:

1. ``_ode_system(t, y, params)`` -- the ODE right-hand side
2. ``_get_initial_conditions()`` -- initial state vector
3. ``_compute_objective(t, y, params)`` -- scalar from ODE solution

----

Full Hierarchy Diagram
======================

.. code-block:: text

    BaseTestFunction
    ├── Fixed: __call__, pure(), batch(), search_space, reset()
    ├── Override: _objective(), _default_search_space(), _batch_objective()
    │
    ├── AlgebraicFunction
    │   ├── Provides: _default_search_space() from default_bounds + n_dim
    │   ├── Override: _objective()
    │   │
    │   └── EngineeringFunction
    │       ├── Fixed: raw_objective(), constraints(), is_feasible(), penalty()
    │       ├── Provides: _objective() = _raw_objective() + penalty()
    │       └── Override: _raw_objective(), _constraints()
    │
    ├── CECFunction
    │   └── Provides: _objective() via shift/rotation transforms
    │
    ├── BBOBFunction
    │   └── Provides: _objective() via COCO transforms
    │
    ├── MachineLearningFunction
    │   ├── Provides: _objective() delegates to _ml_objective()
    │   ├── Provides: _default_search_space() from para_names + *_default attrs
    │   └── Override: _ml_objective()
    │
    └── SimulationFunction
        ├── Provides: _objective() = _run_simulation() + _extract_objective()
        └── Override: _setup_simulation(), _run_simulation(), _extract_objective()
            │
            └── ODESimulationFunction
                ├── Provides: _run_simulation() via scipy solve_ivp
                └── Override: _ode_system(), _get_initial_conditions(),
                              _compute_objective()

----

Adding a New Function
=====================

Algebraic function example
--------------------------

.. code-block:: python

    from surfaces.test_functions.algebraic._base_algebraic_function import AlgebraicFunction

    class MyFunction(AlgebraicFunction):
        _spec = {
            "n_dim": 2,
            "default_bounds": (-5.0, 5.0),
            "convex": True,
            "unimodal": True,
        }

        f_global = 0.0
        x_global = [0.0, 0.0]

        def _objective(self, params):
            x = params["x0"]
            y = params["x1"]
            return x**2 + y**2

        def _batch_objective(self, X):
            xp = get_array_namespace(X)
            return xp.sum(X**2, axis=1)

No need to implement ``_default_search_space`` -- ``AlgebraicFunction`` builds
it automatically from ``_spec["default_bounds"]`` and ``n_dim``.

Engineering function example
----------------------------

.. code-block:: python

    from surfaces.test_functions.algebraic.constrained._base_engineering_function import (
        EngineeringFunction,
    )

    class MyBeam(EngineeringFunction):
        variable_names = ["width", "height"]
        variable_bounds = [(0.1, 10.0), (0.1, 10.0)]

        def _raw_objective(self, params):
            return params["width"] * params["height"]  # minimize area

        def _constraints(self, params):
            # Stress constraint: must be <= 0 for feasibility
            stress = 100.0 / (params["width"] * params["height"]) - 10.0
            return [stress]

ML function example
-------------------

.. code-block:: python

    from surfaces.test_functions.machine_learning._base_machine_learning import (
        MachineLearningFunction,
    )

    class MyClassifier(MachineLearningFunction):
        para_names = ["n_neighbors", "weights"]
        n_neighbors_default = [3, 5, 7, 9, 11]
        weights_default = ["uniform", "distance"]

        def _ml_objective(self, params):
            # Train model, return accuracy
            ...

----

Test Coverage
=============

The template method contracts are verified by:

- ``tests/core/properties/test_template_method.py`` -- checks that all
  discovered classes provide ``_objective`` (directly or via sub-template)
- ``tests/core/properties/test_framework_contracts.py`` -- tests that missing
  overrides produce clear ``NotImplementedError`` messages naming the class
- ``tests/core/properties/test_interface_compliance.py`` -- checks ``search_space``,
  ``_spec``, and other interface requirements across all functions
