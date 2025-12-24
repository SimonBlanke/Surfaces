.. _api_visualization:

=============
Visualization
=============

Functions for visualizing test function landscapes and optimization progress.

.. contents:: On this page
   :local:
   :depth: 2

----

Discovery Functions
===================

Functions to discover which plots work with specific test functions.

.. autofunction:: surfaces.visualize.available_plots

.. autofunction:: surfaces.visualize.check_compatibility

.. autofunction:: surfaces.visualize.plot_info

.. autofunction:: surfaces.visualize.list_all_plots

----

Plot Functions
==============

auto_plot
---------

Automatically selects the best visualization for a function.

.. autofunction:: surfaces.visualize.auto_plot

plot_surface
------------

3D surface plot (2D functions only).

.. autofunction:: surfaces.visualize.plot_surface

plot_contour
------------

2D contour plot (2D functions only).

.. autofunction:: surfaces.visualize.plot_contour

plot_multi_slice
----------------

1D slices through each dimension (any N-D function).

.. autofunction:: surfaces.visualize.plot_multi_slice

plot_convergence
----------------

Best-so-far vs evaluations (requires optimization history).

.. autofunction:: surfaces.visualize.plot_convergence

plot_fitness_distribution
-------------------------

Histogram of sampled values (any N-D function).

.. autofunction:: surfaces.visualize.plot_fitness_distribution

plot_latex
----------

Publication-quality LaTeX/PDF with formula (2D algebraic functions with latex_formula).

.. autofunction:: surfaces.visualize.plot_latex

----

Exceptions
==========

.. autoexception:: surfaces.visualize.VisualizationError
   :show-inheritance:

.. autoexception:: surfaces.visualize.PlotCompatibilityError
   :show-inheritance:

.. autoexception:: surfaces.visualize.MissingDataError
   :show-inheritance:

.. autoexception:: surfaces.visualize.MissingDependencyError
   :show-inheritance:
