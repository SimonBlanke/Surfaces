.. _api_presets:

=======
Presets
=======

.. include:: ../_generated/diagrams/presets_overview.rst

----

Available Presets
=================

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Preset
     - Count
     - Description
   * - ``quick``
     - 5
     - Fast sanity checks during development
   * - ``standard``
     - 15
     - Well-known functions covering diverse landscape types
   * - ``algebraic_2d``
     - 18
     - All 2D algebraic functions
   * - ``algebraic_nd``
     - 5
     - N-dimensional scalable functions
   * - ``bbob``
     - 24
     - Full BBOB/COCO benchmark
   * - ``cec2014``
     - 30
     - CEC 2014 competition functions
   * - ``cec2017``
     - 10
     - CEC 2017 simple functions
   * - ``engineering``
     - 5
     - Constrained engineering design problems

----

Utility Functions
=================

.. autofunction:: surfaces.presets.instantiate

.. autofunction:: surfaces.presets.get

.. autofunction:: surfaces.presets.list_presets
