# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt

color_scale = px.colors.sequential.Jet


def _create_grid(objective_function, search_space):
    def objective_function_np(*args):
        para = {}
        for arg, key in zip(args, search_space.keys()):
            para[key] = arg

        return objective_function(para)

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = objective_function_np(xi, yi)

    print("\n xi \n", xi, xi.shape, "\n")
    print("\n yi \n", yi, yi.shape, "\n")
    print("\n zi \n", zi, zi.shape, "\n")

    return xi, yi, zi


def plotly_surface_new(
    objective_function,
    search_space,
    title="Objective Function Surface",
    width=900,
    height=900,
    contour=False,
):
    if len(search_space) != 2:
        error = "search space must be two dimensional"
        raise Exception(error)

    para_l = list(search_space.keys())

    para1 = para_l[0]
    para2 = para_l[1]

    (x_all, y_all) = search_space.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = []

    for dim_value1 in search_space[para1]:
        zi_sub_dim = []
        for dim_value2 in search_space[para2]:
            para_dict = {
                para1: dim_value1,
                para2: dim_value2,
            }
            zi_sub_dim_values = objective_function(para_dict)
            zi_sub_dim.append(zi_sub_dim_values)

        zi.append(zi_sub_dim)

    zi = np.array(zi).T

    print("\n xi \n", xi, xi.shape, "\n")
    print("\n yi \n", yi, yi.shape, "\n")
    print("\n zi \n", zi, zi.shape, "\n")

    fig = go.Figure(
        data=go.Surface(
            z=zi,
            x=xi,
            y=yi,
            colorscale=color_scale,
        ),
    )

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
            xaxis_title=para1,
            yaxis_title=para2,
            zaxis_title="Metric",
        ),
        width=width,
        height=height,
    )
    return fig


def plotly_surface(
    objective_function,
    search_space,
    title="Objective Function Surface",
    width=900,
    height=900,
    contour=False,
):
    if len(search_space) != 2:
        error = "search space must be two dimensional"
        raise Exception(error)

    xi, yi, zi = _create_grid(objective_function, search_space)

    fig = go.Figure(
        data=go.Surface(
            z=zi,
            x=xi,
            y=yi,
            colorscale=color_scale,
        ),
    )

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
            zaxis_title="Metric",
        ),
        width=width,
        height=height,
    )
    return fig
    # fig.write_image("surface.png")


def plotly_heatmap(
    objective_function,
    search_space,
    title="Objective Function Heatmap",
    width=900,
    height=900,
):
    xi, yi, zi = _create_grid(objective_function, search_space)

    fig = px.imshow(
        img=zi,
        x=search_space["x0"],
        y=search_space["x1"],
        labels=dict(x="X", y="Y", color="Metric"),
        color_continuous_scale=color_scale,
    )
    fig.update_layout(
        title=title,
        width=width,
        height=height,
    )

    return fig


def matplotlib_heatmap(
    objective_function,
    search_space,
    title="Objective Function Heatmap",
    norm=None,
):
    if norm == "color_log":
        norm = mpl.colors.LogNorm()

    xi, yi, zi = _create_grid(objective_function, search_space)

    fig, ax = plt.subplots()
    ax.imshow(
        zi,
        cmap=plt.cm.jet,
        extent=[
            search_space["x0"][0],
            search_space["x0"][-1],
            search_space["x1"][0],
            search_space["x1"][-1],
        ],
        aspect="auto",
        norm=norm,
    )

    fig.tight_layout()
    return plt


def matplotlib_surface(
    objective_function,
    search_space,
    title="Objective Function Surface",
    norm=None,
):
    if norm == "color_log":
        norm = mpl.colors.LogNorm()

    xi, yi, zi = _create_grid(objective_function, search_space)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    ax.plot_surface(
        xi,
        yi,
        zi,
        cmap=plt.cm.jet,
        # linewidth=0,
        # alpha=0.6,
        cstride=1,
        rstride=1,
        antialiased=False,
        shade=False,
        norm=norm,
    )

    pos_ = ax.get_position()
    pos_new = [pos_.x0 + 0.1, pos_.y0 + -0.12, pos_.width, pos_.height]
    ax.set_position(pos_new)

    ax.view_init(30, 15)
    ax.dist = 7.5

    fig.tight_layout()
    return plt
