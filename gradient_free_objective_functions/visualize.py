# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def _create_grid(objective_function, search_space):
    def objective_function_np(*args):
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg

        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    return xi, yi, zi


def plot_surface(
    objective_function,
    search_space,
    title="Objective Function Surface",
    width=900,
    height=900,
    contour=False,
):
    xi, yi, zi = _create_grid(objective_function, search_space)

    fig = go.Figure(data=go.Surface(z=zi, x=xi, y=yi))

    # add a countour plot
    if contour:
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )

    # annotate the plot
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="metric",
        ),
        width=width,
        height=height,
    )
    fig.show()


def plot_heatmap(
    objective_function,
    search_space,
    title="Objective Function Heatmap",
    width=900,
    height=900,
):
    xi, yi, zi = _create_grid(objective_function, search_space)

    fig = px.imshow(zi, labels=dict(x="X", y="Y", color="metric"))
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )
    fig.show()
