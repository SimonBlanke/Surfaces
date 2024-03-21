# Author: Simon Blanke
# Email: simon.blanke@yahoo.com
# License: MIT License


import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm

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

    return xi, yi, zi


def plotly_surface_new(
    objective_function,
    search_space,
    title="Objective Function Surface",
    width=900,
    height=900,
    contour=False,
):
    if len(search_space) < 2:
        error = "search space must be at least two dimensional"
        raise Exception(error)
    elif len(search_space) > 2:
        search_space_2d = {}
        para_dict_set_values = {}
        for para_name, dim_values in search_space.items():
            if len(dim_values) == 1:
                para_dict_set_values[para_name] = dim_values[0]
            else:
                search_space_2d[para_name] = dim_values
    else:
        search_space_2d = search_space
        para_dict_set_values = {}

    para_l = list(search_space_2d.keys())

    para1 = para_l[0]
    para2 = para_l[1]

    (x_all, y_all) = search_space_2d.values()
    xi, yi = np.meshgrid(x_all, y_all)
    zi = []

    pbar_total = len(search_space_2d[para1]) * len(search_space_2d[para2])
    pbar = tqdm(total=pbar_total)
    for dim_value1 in search_space_2d[para1]:
        zi_sub_dim = []
        for dim_value2 in search_space_2d[para2]:
            para_dict_2d = {
                para1: dim_value1,
                para2: dim_value2,
            }
            para_dict = {**para_dict_2d, **para_dict_set_values}

            zi_sub_dim_values = objective_function(para_dict)
            zi_sub_dim.append(zi_sub_dim_values)
            pbar.update(1)

        zi.append(zi_sub_dim)

    zi = np.array(zi).T
    pbar.close()

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
