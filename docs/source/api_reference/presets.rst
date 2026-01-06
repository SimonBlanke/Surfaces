.. _api_collection:

==========
Collection
==========

.. include:: ../_generated/diagrams/presets_overview.rst

----

Available Suites
================

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Suite
     - Count
     - Description
   * - ``collection.quick``
     - 5
     - Fast sanity checks during development
   * - ``collection.standard``
     - 15
     - Well-known functions covering diverse landscape types
   * - ``collection.algebraic_2d``
     - 18
     - All 2D algebraic functions
   * - ``collection.algebraic_nd``
     - 5
     - N-dimensional scalable functions
   * - ``collection.bbob``
     - 24
     - Full BBOB/COCO benchmark
   * - ``collection.cec2014``
     - 30
     - CEC 2014 competition functions
   * - ``collection.cec2017``
     - 10
     - CEC 2017 simple functions
   * - ``collection.engineering``
     - 5
     - Constrained engineering design problems

----

Collection Class
================

.. autoclass:: surfaces.collection.Collection
   :members:
   :undoc-members:

----

Singleton Instance
==================

.. py:data:: surfaces.collection.collection

   Pre-instantiated singleton containing all test functions.
   Access predefined suites via properties like ``collection.quick``,
   ``collection.standard``, etc.

   **Example:**

   .. code-block:: python

      from surfaces import collection

      # Iterate over all functions
      for func_cls in collection:
          print(func_cls.__name__)

      # Access predefined suites
      quick_funcs = collection.quick.instantiate(n_dim=10)

      # Filter and search
      unimodal = collection.filter(unimodal=True)
      rastrigin = collection.search("rastrigin")

----

Utility Functions
=================

.. autofunction:: surfaces.collection.instantiate
