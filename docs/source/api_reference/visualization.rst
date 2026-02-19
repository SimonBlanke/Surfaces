.. _api_visualization:

=============
Visualization
=============

.. include:: ../_generated/diagrams/visualization_overview.rst

----

Discovery Functions
===================

Functions to discover which plots work with specific test functions.

.. autofunction:: surfaces._visualize.available_plots

.. autofunction:: surfaces._visualize.check_compatibility

.. autofunction:: surfaces._visualize.plot_info

.. autofunction:: surfaces._visualize.list_all_plots

----

Plot Functions
==============

auto_plot
---------

Automatically selects the best visualization for a function.

.. autofunction:: surfaces._visualize.auto_plot

plot_surface
------------

3D surface plot (2D functions only).

.. autofunction:: surfaces._visualize.plot_surface

plot_contour
------------

2D contour plot (2D functions only).

.. autofunction:: surfaces._visualize.plot_contour

plot_multi_slice
----------------

1D slices through each dimension (any N-D function).

.. autofunction:: surfaces._visualize.plot_multi_slice

plot_convergence
----------------

Best-so-far vs evaluations (requires optimization history).

.. autofunction:: surfaces._visualize.plot_convergence

plot_fitness_distribution
-------------------------

Histogram of sampled values (any N-D function).

.. autofunction:: surfaces._visualize.plot_fitness_distribution

plot_latex
----------

Publication-quality LaTeX/PDF with formula (2D algebraic functions with latex_formula).

.. autofunction:: surfaces._visualize.plot_latex

----

Exceptions
==========

.. autoexception:: surfaces._visualize.VisualizationError
   :show-inheritance:

.. autoexception:: surfaces._visualize.PlotCompatibilityError
   :show-inheritance:

.. autoexception:: surfaces._visualize.MissingDataError
   :show-inheritance:

.. autoexception:: surfaces._visualize.MissingDependencyError
   :show-inheritance:
