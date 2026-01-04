.. _example_surface_plots:

=============
Surface Plots
=============

3D surface visualizations of test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic Surface Plot
==================

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    from surfaces.visualization import plot_surface

    func = AckleyFunction()

    # Create 3D surface plot
    fig = plot_surface(func, title="Ackley Function")
    fig.show()

----

Customizing the Plot
====================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from surfaces.visualization import plot_surface

    func = RastriginFunction(n_dim=2)

    fig = plot_surface(
        func,
        title="Rastrigin Function",
        resolution=100,  # Grid resolution
    )
    fig.show()

----

Saving Plots
============

.. code-block:: python

    from surfaces.test_functions import HimmelblausFunction
    from surfaces.visualization import plot_surface

    func = HimmelblausFunction()
    fig = plot_surface(func)

    # Save as interactive HTML
    fig.write_html("himmelblau_surface.html")

    # Save as static image (requires kaleido)
    fig.write_image("himmelblau_surface.png")

----

Multiple Functions
==================

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        AckleyFunction,
        RosenbrockFunction,
    )
    from surfaces.visualization import plot_surface

    functions = [
        SphereFunction(n_dim=2),
        RastriginFunction(n_dim=2),
        AckleyFunction(),
        RosenbrockFunction(n_dim=2),
    ]

    for func in functions:
        name = func.__class__.__name__
        fig = plot_surface(func, title=name)
        fig.write_html(f"{name.lower()}.html")
        print(f"Saved {name.lower()}.html")
