.. _examples_visualization:

=============
Visualization
=============

Examples of visualizing test function landscapes.

Basic Surface Plot
==================

.. code-block:: python

    from surfaces.test_functions import AckleyFunction
    import numpy as np
    import plotly.graph_objects as go

    func = AckleyFunction()

    # Create grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

    # Create plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    fig.update_layout(
        title='Ackley Function',
        scene=dict(
            xaxis_title='x0',
            yaxis_title='x1',
            zaxis_title='f(x0, x1)'
        )
    )
    fig.write_html("ackley_surface.html")
    print("Saved to ackley_surface.html")

Contour Plot
============

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    import numpy as np
    import plotly.graph_objects as go

    func = RosenbrockFunction(n_dim=2)

    # Create grid
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 3, 200)
    X, Y = np.meshgrid(x, y)

    # Evaluate function
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

    # Create contour plot
    fig = go.Figure(data=go.Contour(
        x=x, y=y, z=Z,
        colorscale='RdBu',
        contours=dict(
            start=0,
            end=100,
            size=5
        )
    ))
    fig.update_layout(
        title='Rosenbrock Function Contours',
        xaxis_title='x0',
        yaxis_title='x1'
    )
    fig.write_html("rosenbrock_contour.html")

Comparing Multiple Functions
============================

.. code-block:: python

    from surfaces.test_functions import (
        SphereFunction,
        AckleyFunction,
        RastriginFunction,
        HimmelblausFunction
    )
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    functions = [
        ('Sphere', SphereFunction(n_dim=2)),
        ('Ackley', AckleyFunction()),
        ('Rastrigin', RastriginFunction(n_dim=2)),
        ('Himmelblau', HimmelblausFunction())
    ]

    # Create grid
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[name for name, _ in functions],
        specs=[[{'type': 'surface'}] * 2] * 2
    )

    for idx, (name, func) in enumerate(functions):
        row, col = idx // 2 + 1, idx % 2 + 1

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, colorscale='Viridis', showscale=False),
            row=row, col=col
        )

    fig.update_layout(height=800, width=1000, title_text="Test Function Comparison")
    fig.write_html("function_comparison.html")

Visualizing Optimization Path
=============================

.. code-block:: python

    from surfaces.test_functions import RosenbrockFunction
    from scipy.optimize import minimize
    import numpy as np
    import plotly.graph_objects as go

    func = RosenbrockFunction(n_dim=2)
    objective, bounds, x0 = func.to_scipy()

    # Track path
    path = [x0.copy()]

    def callback(x):
        path.append(x.copy())

    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', callback=callback)
    path = np.array(path)

    # Create contour background
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

    fig = go.Figure()

    # Add contour
    fig.add_trace(go.Contour(
        x=x, y=y, z=Z,
        colorscale='Blues',
        opacity=0.7,
        showscale=False
    ))

    # Add optimization path
    fig.add_trace(go.Scatter(
        x=path[:, 0],
        y=path[:, 1],
        mode='lines+markers',
        marker=dict(size=8, color='red'),
        line=dict(color='red', width=2),
        name='Optimization Path'
    ))

    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[path[0, 0]], y=[path[0, 1]],
        mode='markers',
        marker=dict(size=15, color='green', symbol='star'),
        name='Start'
    ))
    fig.add_trace(go.Scatter(
        x=[path[-1, 0]], y=[path[-1, 1]],
        mode='markers',
        marker=dict(size=15, color='gold', symbol='star'),
        name='End'
    ))

    fig.update_layout(
        title='L-BFGS-B Optimization Path on Rosenbrock',
        xaxis_title='x0',
        yaxis_title='x1'
    )
    fig.write_html("optimization_path.html")

Heatmap with Logarithmic Scale
==============================

.. code-block:: python

    from surfaces.test_functions import RastriginFunction
    import numpy as np
    import plotly.graph_objects as go

    func = RastriginFunction(n_dim=2)

    x = np.linspace(-5.12, 5.12, 200)
    y = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func({"x0": X[i, j], "x1": Y[i, j]})

    # Log scale for better visualization
    Z_log = np.log10(Z + 1)

    fig = go.Figure(data=go.Heatmap(
        x=x, y=y, z=Z_log,
        colorscale='Plasma',
        colorbar=dict(title='log10(f+1)')
    ))

    fig.update_layout(
        title='Rastrigin Function (Log Scale)',
        xaxis_title='x0',
        yaxis_title='x1'
    )
    fig.write_html("rastrigin_heatmap.html")
