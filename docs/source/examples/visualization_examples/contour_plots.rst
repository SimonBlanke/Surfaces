.. _example_contour_plots:

=============
Contour Plots
=============

2D contour visualizations of test functions.

.. contents:: On this page
   :local:
   :depth: 2

----

Basic Contour Plot
==================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    from surfaces.visualization import plot_contour

    func = RastriginFunction(n_dim=2)

    # Create contour plot
    fig = plot_contour(func, title="Rastrigin Contour")
    fig.show()

----

Customizing Contours
====================

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from surfaces.visualization import plot_contour

    func = RosenbrockFunction(n_dim=2)

    fig = plot_contour(
        func,
        title="Rosenbrock Function",
        resolution=200,  # Higher resolution
    )
    fig.show()

----

Comparing Landscapes
====================

.. code-block:: python

    """Compare landscapes of different functions."""

    from surfaces.test_functions import (
        SphereFunction,
        RastriginFunction,
        HimmelblausFunction,
    )
    from surfaces.visualization import plot_contour

    functions = [
        ("Sphere (unimodal)", SphereFunction(n_dim=2)),
        ("Rastrigin (multimodal)", RastriginFunction(n_dim=2)),
        ("Himmelblau (4 minima)", HimmelblausFunction()),
    ]

    for title, func in functions:
        fig = plot_contour(func, title=title)
        filename = title.split()[0].lower() + "_contour.html"
        fig.write_html(filename)
        print(f"Saved {filename}")

----

Matplotlib Backend
==================

.. code-block:: python

    """Use matplotlib for publication-quality figures."""

    import matplotlib.pyplot as plt
    import numpy as np
    from surfaces.test_functions import BealeFunction

    func = BealeFunction()
    space = func.search_space()

    # Create grid
    x = np.linspace(space['x0'].min(), space['x0'].max(), 100)
    y = np.linspace(space['x1'].min(), space['x1'].max(), 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

    # Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='f(x, y)')
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title("Beale Function")
    plt.savefig("beale_contour.pdf", dpi=300)
    plt.show()
