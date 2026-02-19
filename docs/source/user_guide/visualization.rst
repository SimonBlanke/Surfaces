.. _user_guide_visualization:

=============
Visualization
=============

Surfaces provides built-in visualization through the ``.plot`` accessor
on every test function. All plots use Plotly and return interactive
``plotly.graph_objects.Figure`` objects.

.. code-block:: python

    from surfaces.test_functions.algebraic import AckleyFunction

    func = AckleyFunction()
    fig = func.plot.surface()
    fig.show()


Plot Compatibility
==================

Not every plot type works with every function. Availability depends
primarily on the number of dimensions. The table below is
auto-generated from the actual ``func.plot.available()`` output.

.. include:: /_generated/diagrams/plot_compatibility.rst


Available Plot Types
====================

Surface
-------

Interactive 3D surface showing the objective landscape.
Requires exactly 2 plotted dimensions.

.. code-block:: python

    func = AckleyFunction()
    fig = func.plot.surface()

Contour
-------

2D contour plot with isolines of equal objective value.
Requires exactly 2 plotted dimensions.

.. code-block:: python

    fig = func.plot.contour()

Heatmap
-------

Color-coded grid of objective values. Requires exactly 2 plotted
dimensions.

.. code-block:: python

    fig = func.plot.heatmap()

Multi-Slice
-----------

1D slices through each dimension while all others are fixed.
Works with any number of dimensions.

.. code-block:: python

    from surfaces.test_functions.algebraic import SphereFunction

    func = SphereFunction(n_dim=5)
    fig = func.plot.multi_slice()

Fitness Distribution
--------------------

Histogram of objective values from random sampling across the
search space. Works with any number of dimensions.

.. code-block:: python

    fig = func.plot.fitness_distribution()

Convergence
-----------

Best-so-far objective value vs. evaluation number. Requires
evaluation history (either from ``func.search_data`` or passed
explicitly).

.. code-block:: python

    # After running evaluations:
    fig = func.plot.convergence()

    # Or with explicit history:
    fig = func.plot.convergence(history=my_history)

    # Or via chaining:
    fig = func.plot.with_history(my_history).convergence()

LaTeX/PDF
---------

Publication-quality output with pgfplots surface and formula.
Only available for 2D algebraic functions that have a
``latex_formula`` attribute.

.. code-block:: python

    pdf_path = func.plot.latex()
    pdf_path = func.plot.latex(output_path="my_plot.pdf")


Discovering Available Plots
===========================

Use ``func.plot.available()`` to check which plots work for a
given function instance:

.. code-block:: python

    func = SphereFunction(n_dim=2)
    func.plot.available()
    # ['surface', 'contour', 'heatmap', 'multi_slice', 'fitness_distribution', 'latex']

    func = SphereFunction(n_dim=5)
    func.plot.available()
    # ['multi_slice', 'fitness_distribution']


N-D Functions: Selecting Dimensions
====================================

For functions with more than 2 dimensions, use the ``params`` dict
to select which dimensions to plot and fix the rest:

.. code-block:: python

    func = SphereFunction(n_dim=5)

    # Plot x0 vs x2, fix others at defaults
    fig = func.plot.surface(params={"x0": ..., "x2": ...})

    # Custom ranges and fixed values
    fig = func.plot.surface(params={
        "x0": (-2, 2),
        "x2": (-1, 1),
        "x1": 0.0,
        "x3": 0.5,
        "x4": -1.0,
    })


Customization
=============

Resolution
----------

Control the grid resolution (higher = more detail, slower):

.. code-block:: python

    fig = func.plot.surface(resolution=200)
    fig = func.plot.contour(resolution=30)

Plotly kwargs
-------------

Additional keyword arguments are forwarded to the underlying Plotly
calls:

.. code-block:: python

    fig = func.plot.surface(colorscale="Plasma")
    fig = func.plot.heatmap(colorscale="RdBu")


Saving Plots
=============

.. code-block:: python

    # Interactive HTML
    fig.write_html("function_surface.html")

    # Static image (requires kaleido)
    fig.write_image("function_surface.png")
    fig.write_image("function_surface.svg")
