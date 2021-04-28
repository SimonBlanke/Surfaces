# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import plotly.graph_objects as go


def plot_surface(
    objective_function,
    search_space,
    title="Objective Function Surface",
    width=700,
    height=700,
    contour=False,
):
    def objective_function_np(*args):
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg

        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

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
            xaxis_title="x1",
            yaxis_title="x2",
            zaxis_title="metric",
        ),
        width=width,
        height=height,
    )

    fig.show()