.. _installation_visualization:

=============
Visualization
=============

Visualization tools help you understand test function landscapes
through surface plots, contour plots, and interactive visualizations.

----

Installation
============

.. code-block:: bash

    pip install surfaces[viz]

This installs:

- **plotly**: Interactive plots and 3D surfaces
- **matplotlib**: Static plots and publication-quality figures

----

What You Get
============

Surface Plots
-------------

3D visualization of 2D test functions:

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    from surfaces.visualization import plot_surface

    func = AckleyFunction()
    fig = plot_surface(func)
    fig.show()

Contour Plots
-------------

2D contour visualization:

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from surfaces.visualization import plot_contour

    func = RastriginFunction(n_dim=2)
    fig = plot_contour(func)
    fig.show()

----

Interactive Features
====================

Plotly-based visualizations are interactive:

- **Rotate**: Click and drag to rotate 3D surfaces
- **Zoom**: Scroll to zoom in/out
- **Pan**: Shift + drag to pan
- **Hover**: See exact values at any point

----

Usage Examples
==============

Basic Surface Plot
------------------

.. code-block:: python

    from surfaces.test_functions import HimmelblausFunction
    from surfaces.visualization import plot_surface

    func = HimmelblausFunction()

    # Create interactive 3D surface
    fig = plot_surface(
        func,
        title="Himmelblau's Function",
        resolution=100
    )
    fig.show()

Save to File
------------

.. code-block:: python

    from surfaces.visualization import plot_surface, plot_contour

    # Save as HTML (interactive)
    fig = plot_surface(func)
    fig.write_html("surface.html")

    # Save as PNG (static)
    fig.write_image("surface.png")

Multiple Functions
------------------

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        AckleyFunction
    )
    from surfaces.visualization import plot_surface

    functions = [
        SphereFunction(n_dim=2),
        RastriginFunction(n_dim=2),
        AckleyFunction()
    ]

    for func in functions:
        fig = plot_surface(func, title=func.__class__.__name__)
        fig.write_html(f"{func.__class__.__name__.lower()}.html")

----

Matplotlib Backend
==================

For publication-quality static figures:

.. code-block:: python

    import matplotlib.pyplot as plt
    from surfaces.test_functions import BealeFunction
    from surfaces.visualization import plot_contour_matplotlib

    func = BealeFunction()

    fig, ax = plot_contour_matplotlib(func)
    plt.savefig("beale_contour.pdf", dpi=300)

----

Visualization in Documentation
==============================

The Surfaces documentation includes a :ref:`function_gallery` with
pre-generated visualizations of all 2D test functions.

----

Dependencies Note
=================

Plotly
------

Plotly is the primary visualization library. It provides:

- Interactive 3D surfaces
- WebGL rendering for performance
- HTML export for sharing

Matplotlib
----------

Matplotlib is optional but useful for:

- Publication-quality figures
- PDF/EPS export
- Custom styling

Kaleido
-------

For static image export (PNG, PDF), you may need kaleido:

.. code-block:: bash

    pip install kaleido

----

Next Steps
==========

- :doc:`/user_guide/visualization` - Visualization guide
- :doc:`/examples/visualization` - Visualization examples
- :doc:`/api_reference/visualization` - API reference
