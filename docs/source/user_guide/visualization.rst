.. _user_guide_visualization:

=============
Visualization
=============

Surfaces provides built-in visualization tools for exploring test function
landscapes using Plotly.

Overview
========

Visualization helps you understand:

- Function topology (valleys, peaks, plateaus)
- Global and local optima locations
- Search space characteristics
- Algorithm behavior

The visualization module provides surface plots and heatmaps for
2D functions.

Basic Surface Plot
==================

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    from surfaces import visualize

    # Create a 2D function
    func = AckleyFunction()

    # Create a surface plot
    fig = visualize.surface_plot(func)
    fig.show()

This creates an interactive 3D surface plot using Plotly.

Heatmap Plot
============

For a top-down view:

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from surfaces import visualize

    func = RosenbrockFunction(n_dim=2)

    # Create a heatmap
    fig = visualize.heatmap(func)
    fig.show()

Customizing Plots
=================

Resolution
----------

Control the grid resolution:

.. code-block:: python

    # Higher resolution (slower)
    fig = visualize.surface_plot(func, resolution=200)

    # Lower resolution (faster)
    fig = visualize.surface_plot(func, resolution=50)

Custom Bounds
-------------

Zoom in on specific regions:

.. code-block:: python

    fig = visualize.surface_plot(
        func,
        x_range=(-2, 2),
        y_range=(-2, 2)
    )

Color Scales
------------

Change the color scheme:

.. code-block:: python

    fig = visualize.surface_plot(func, colorscale='Viridis')
    fig = visualize.heatmap(func, colorscale='RdBu')

Available colorscales include: 'Viridis', 'Plasma', 'Inferno', 'Magma',
'Cividis', 'RdBu', 'Blues', 'Greens', etc.

Visualizing N-D Functions
=========================

For N-dimensional functions, visualization shows a 2D slice:

.. code-block:: python

    from surfaces.test_functions import SphereFunction

    # Create a 5D function
    func = SphereFunction(n_dim=5)

    # Visualize x0 vs x1 (other dimensions fixed at 0)
    fig = visualize.surface_plot(
        func,
        dims=('x0', 'x1'),
        fixed_values={'x2': 0, 'x3': 0, 'x4': 0}
    )

Comparing Functions
===================

Visualize multiple functions side by side:

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        AckleyFunction,
        RastriginFunction
    )
    from surfaces import visualize
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    functions = [
        ('Sphere', SphereFunction(n_dim=2)),
        ('Ackley', AckleyFunction()),
        ('Rastrigin', RastriginFunction(n_dim=2))
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[name for name, _ in functions],
        specs=[[{'type': 'surface'}] * 3]
    )

    for i, (name, func) in enumerate(functions, 1):
        surface = visualize.get_surface_data(func)
        fig.add_trace(surface, row=1, col=i)

    fig.update_layout(height=400, width=1200)
    fig.show()

Saving Plots
============

Save visualizations to files:

.. code-block:: python

    # Save as HTML (interactive)
    fig.write_html("function_surface.html")

    # Save as PNG (static)
    fig.write_image("function_surface.png")

    # Save as SVG (vector)
    fig.write_image("function_surface.svg")

Note: Saving to image formats requires the kaleido package:

.. code-block:: bash

    pip install kaleido

Understanding Function Landscapes
=================================

Different functions have different characteristics visible in plots:

Unimodal Functions
------------------

Single global minimum, smooth gradients:

- Sphere: Simple bowl shape
- Matyas: Elliptical bowl

Multimodal Functions
--------------------

Multiple local minima:

- Ackley: Central funnel with flat outer region
- Rastrigin: Regular grid of local minima
- Himmelblau: Four symmetric minima

Valley Functions
----------------

Narrow valleys leading to optimum:

- Rosenbrock: Banana-shaped valley
- Beale: Curved valley

Tips for Interpretation
=======================

1. **Color intensity**: Darker colors typically indicate lower values
   (better for minimization)

2. **Surface smoothness**: Rough surfaces indicate multimodality

3. **Valley width**: Narrow valleys are harder for optimizers

4. **Plateau regions**: Flat areas provide no gradient information

5. **Scale**: Pay attention to the z-axis scale when comparing functions
