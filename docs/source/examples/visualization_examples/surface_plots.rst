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

    import os
    from surfaces.test_functions.algebraic import AckleyFunction
    from surfaces._visualize import plot_surface

    func = AckleyFunction()

    # Create 3D surface plot
    fig = plot_surface(func, title="Ackley Function")

    if not os.environ.get("SURFACES_TESTING"):
        fig.show()

----

Customizing the Plot
====================

.. code-block:: python

    import os
    from surfaces.test_functions.algebraic import RastriginFunction
    from surfaces._visualize import plot_surface

    func = RastriginFunction(n_dim=2)

    fig = plot_surface(
        func,
        title="Rastrigin Function",
        resolution=100,  # Grid resolution
    )

    if not os.environ.get("SURFACES_TESTING"):
        fig.show()

----

Saving Plots
============

.. code-block:: python

    import os
    from surfaces.test_functions.algebraic import HimmelblausFunction
    from surfaces._visualize import plot_surface

    func = HimmelblausFunction()
    fig = plot_surface(func)

    # Save as interactive HTML (always works)
    # fig.write_html("himmelblau_surface.html")

    print("Plot created successfully!")

----

Multiple Functions
==================

.. code-block:: python

    import os
    from surfaces.test_functions.algebraic import (
        SphereFunction,
        RastriginFunction,
        AckleyFunction,
        RosenbrockFunction,
    )
    from surfaces._visualize import plot_surface

    functions = [
        SphereFunction(n_dim=2),
        RastriginFunction(n_dim=2),
        AckleyFunction(),
        RosenbrockFunction(n_dim=2),
    ]

    for func in functions:
        name = func.__class__.__name__
        fig = plot_surface(func, title=name)
        print(f"Created plot for {name}")
