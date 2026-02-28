.. _api_namespace_plot:

=====
.plot
=====

The ``.plot`` namespace provides visualization methods directly on test function
instances via the ``func.plot.surface()``, ``func.plot.contour()`` pattern.

PlotAccessor (built-in test functions)
======================================

Accessor object returned by ``func.plot`` on built-in test functions.

.. autoclass:: surfaces._visualize.PlotAccessor
   :members:
   :show-inheritance:

----

PlotNamespace (custom test functions)
=====================================

Accessor object returned by ``func.plot`` on :class:`~surfaces.custom_test_function.CustomTestFunction`.

.. autoclass:: surfaces.custom_test_function._namespaces.PlotNamespace
   :members:
   :show-inheritance:
