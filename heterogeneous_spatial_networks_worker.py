"""
By Jason Mulderrig
https://github.com/jasonmulderrig/heterogeneous-spatial-networks/blob/main/heterogeneous_spatial_networks_worker.py
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from heterogeneous_spatial_networks_funcs import (
    filename_str,
    filepath_str,
    voronoi_L,
    delaunay_L,
    swidt_L,
    aelp_L,
    node_seeding,
    voronoi_network_topology_initialization,
    delaunay_network_topology_initialization,
    swidt_network_topology_initialization,
    aelp_network_topology_initialization,
    swidt_network_edge_pruning_procedure,
    swidt_network_k_counts,
    swidt_network_h_counts,
    swidt_network_l_edges,
    swidt_network_l_nrmlzd_edges,
    swidt_network_l_cmpnts_edges,
    swidt_network_l_cmpnts_nrmlzd_edges
)

def run_voronoi_L(args):
    voronoi_L(*args)

def run_delaunay_L(args):
    delaunay_L(*args)

def run_swidt_L(args):
    swidt_L(*args)

def run_aelp_L(args):
    aelp_L(*args)

def run_node_seeding(args):
    node_seeding(*args)

def run_voronoi_network_topology_initialization(args):
    voronoi_network_topology_initialization(*args)

def run_delaunay_network_topology_initialization(args):
    delaunay_network_topology_initialization(*args)

def run_swidt_network_topology_initialization(args):
    swidt_network_topology_initialization(*args)

def run_aelp_network_topology_initialization(args):
    aelp_network_topology_initialization(*args)

def run_swidt_network_edge_pruning_procedure(args):
    swidt_network_edge_pruning_procedure(*args)

def run_swidt_network_k_counts(args):
    swidt_network_k_counts(*args)

def run_swidt_network_h_counts(args):
    swidt_network_h_counts(*args)

def run_swidt_network_l_edges(args):
    swidt_network_l_edges(*args)

def run_swidt_network_l_nrmlzd_edges(args):
    swidt_network_l_nrmlzd_edges(*args)

def run_swidt_network_l_cmpnts_edges(args):
    swidt_network_l_cmpnts_edges(*args)

def run_swidt_network_l_cmpnts_nrmlzd_edges(args):
    swidt_network_l_cmpnts_nrmlzd_edges(*args)

def dim_2_network_topology_axes_formatter(
        ax: plt.axes,
        core_square: np.ndarray,
        core_square_color: str,
        core_square_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    ax.plot(
        core_square[:, 0], core_square[:, 1],
        color=core_square_color, linewidth=core_square_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax

def dim_3_network_topology_axes_formatter(
        ax: plt.axes,
        core_cube: np.ndarray,
        core_cube_color: str,
        core_cube_linewidth: float,
        xlim: np.ndarray,
        ylim: np.ndarray,
        zlim: np.ndarray,
        xticks: np.ndarray,
        yticks: np.ndarray,
        zticks: np.ndarray,
        xlabel: str,
        ylabel: str,
        zlabel: str,
        grid_alpha: float,
        grid_zorder: int) -> plt.axes:
    for face in np.arange(6):
        ax.plot(
            core_cube[face, :, 0], core_cube[face, :, 1], core_cube[face, :, 2],
            color=core_cube_color, linewidth=core_cube_linewidth)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_zticks(zticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.grid(True, alpha=grid_alpha, zorder=grid_zorder)
    return ax

def voronoi_network_topology_synthesis_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        tsslltd_core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        config: int,
        params_arr: np.ndarray) -> None:
    # Import functions
    from scipy.spatial import Voronoi
    from heterogeneous_spatial_networks_funcs import tessellation_protocol

    # Identification of the sample value for the desired network
    sample = int(np.where((params_arr == (dim, b, n, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Fundamental graph constituents filenames
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    input_core_x_filename = filename_prefix + "-input_core_x" + ".dat"
    input_core_y_filename = filename_prefix + "-input_core_y" + ".dat"
    input_core_z_filename = filename_prefix + "-input_core_z" + ".dat"
    # Plots filenames
    input_core_node_coords_filename = (
        filename_prefix + "-input_core_node_coords" + ".png"
    )
    core_node_coords_filename = (
        filename_prefix + "-core_node_coords" + ".png"
    )
    core_pb_node_coords_filename = (
        filename_prefix + "-core_pb_node_coords" + ".png"
    )
    core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_topology_synthesis" + ".png"
    )
    core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_colored_topology_synthesis" + ".png"
    )
    conn_graph_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_topology_synthesis" + ".png"
    )
    conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_colored_topology_synthesis"
        + ".png"
    )
    tsslltd_input_core_node_coords_filename = (
        filename_prefix + "-tsslltd_input_core_node_coords" + ".png"
    )
    tsslltd_input_core_voronoi_filename = (
        filename_prefix + "-tsslltd_input_core_voronoi" + ".png"
    )
    tsslltd_input_core_voronoi_zoomed_in_filename = (
        filename_prefix + "-tsslltd_input_core_voronoi_zoomed_in" + ".png"
    )
    tsslltd_core_voronoi_filename = (
        filename_prefix + "-tsslltd_core_voronoi" + ".png"
    )
    tsslltd_core_voronoi_zoomed_in_filename = (
        filename_prefix + "-tsslltd_core_voronoi_zoomed_in" + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])
    input_core_x = np.loadtxt(input_core_x_filename)
    input_core_y = np.loadtxt(input_core_y_filename)
    input_core_z = np.asarray([])
    # Tessellated core node coordinates
    input_tsslltd_core_x = input_core_x.copy()
    input_tsslltd_core_y = input_core_y.copy()
    input_tsslltd_core_z = np.asarray([])

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot preformatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    tsslltd_core_tick_inc = tsslltd_core_tick_inc_prefactor * L

    min_core = -plt_pad
    max_core = L + plt_pad
    min_tsslltd_core = -L - plt_pad
    max_tsslltd_core = 2 * L + plt_pad
    
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1
    tsslltd_core_tick_steps = (
        int(np.around((max_tsslltd_core-min_tsslltd_core)/tsslltd_core_tick_inc))
        + 1
    )

    core_xlim = np.asarray([min_core, max_core])
    core_ylim = np.asarray([min_core, max_core])
    core_zlim = np.asarray([min_core, max_core])
    tsslltd_core_xlim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_ylim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_zlim = np.asarray([min_tsslltd_core, max_tsslltd_core])

    core_xticks = np.linspace(min_core, max_core, core_tick_steps)
    core_yticks = np.linspace(min_core, max_core, core_tick_steps)
    core_zticks = np.linspace(min_core, max_core, core_tick_steps)
    tsslltd_core_xticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_yticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_zticks = np.linspace(
            min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and plot preformatting
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]
        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_2_tessellation,
            dim_2_core_pb_edge_identification
        )
        from scipy.spatial import voronoi_plot_2d
        
        # Plot input core nodes
        fig, ax = plt.subplots()
        ax.scatter(input_core_x, input_core_y, s=1.5, marker=".", color="red")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(input_core_node_coords_filename)
        plt.close()

        # Tessellated Delaunay triangulation
        dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
        for tsslltn in range(dim_2_tsslltn_num):
            x_tsslltn = dim_2_tsslltn[tsslltn, 0]
            y_tsslltn = dim_2_tsslltn[tsslltn, 1]
            if (x_tsslltn == 0) and (y_tsslltn == 0): continue
            else:
                core_tsslltn_x, core_tsslltn_y = dim_2_tessellation(
                    L, input_core_x, input_core_y, x_tsslltn, y_tsslltn)
                input_tsslltd_core_x = np.concatenate((input_tsslltd_core_x, core_tsslltn_x))
                input_tsslltd_core_y = np.concatenate((input_tsslltd_core_y, core_tsslltn_y))
        
        del core_tsslltn_x, core_tsslltn_y

        tsslltd_input_core = np.column_stack((input_tsslltd_core_x, input_tsslltd_core_y))

        tsslltd_core_voronoi = Voronoi(tsslltd_input_core)

        del tsslltd_input_core

        vertices = tsslltd_core_voronoi.vertices
        
        vertices_x = vertices[:, 0]
        vertices_y = vertices[:, 1]

        # Plot tessellated input core nodes
        fig, ax = plt.subplots()
        ax.scatter(
            input_tsslltd_core_x, input_tsslltd_core_y, s=1.5, marker=".",
            color="red")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_input_core_node_coords_filename)
        plt.close()

        # Plot Voronoi-tessellated diagram with tessellated input core
        # nodes
        fig, ax = plt.subplots()
        fig = voronoi_plot_2d(
            tsslltd_core_voronoi, ax=ax, show_points=False, show_vertices=False,
            line_colors="tab:blue", line_width=0.75)
        ax.scatter(
            input_tsslltd_core_x, input_tsslltd_core_y, s=1.5, marker=".",
            color="red")
        ax.scatter(vertices_x, vertices_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_input_core_voronoi_filename)
        plt.close()

        fig, ax = plt.subplots()
        fig = voronoi_plot_2d(
            tsslltd_core_voronoi, ax=ax, show_points=False, show_vertices=False,
            line_colors="tab:blue", line_width=0.75)
        ax.scatter(
            input_tsslltd_core_x, input_tsslltd_core_y, s=1.5, marker=".",
            color="red")
        ax.scatter(vertices_x, vertices_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_input_core_voronoi_zoomed_in_filename)
        plt.close()

        # Plot Voronoi-tessellated diagram without tessellated input
        # core nodes
        fig, ax = plt.subplots()
        fig = voronoi_plot_2d(
            tsslltd_core_voronoi, ax=ax, show_points=False, show_vertices=False,
            line_colors="tab:blue", line_width=0.75)
        ax.scatter(vertices_x, vertices_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_voronoi_filename)
        plt.close()

        fig, ax = plt.subplots()
        fig = voronoi_plot_2d(
            tsslltd_core_voronoi, ax=ax, show_points=False, show_vertices=False,
            line_colors="tab:blue", line_width=0.75)
        ax.scatter(vertices_x, vertices_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_voronoi_zoomed_in_filename)
        plt.close()
        
        # Extract two-dimensional periodic boundary node coordinates
        pb_x = []
        pb_y = []
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            pb_x.append(pb_node_0_x)
            pb_x.append(pb_node_1_x)
            pb_y.append(pb_node_0_y)
            pb_y.append(pb_node_1_y)
        pb_x = np.asarray(pb_x)
        pb_y = np.asarray(pb_y)
        pb_n = np.shape(pb_x)[0]

        # Plot core nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_node_coords_filename)
        plt.close()

        # Plot core and periodic boundary nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax.scatter(pb_x, pb_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_node_coords_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        # # Import three-dimension specific functions
        # from heterogeneous_spatial_networks_funcs import (
        #     dim_3_tessellation,
        #     dim_3_core_pb_edge_identification
        # )
        
        # # Load fundamental z-dimensional graph constituents
        # core_z = np.loadtxt(core_z_filename)
        # # Tessellated core node z-coordinates
        # tsslltd_core_z = core_z.copy()

        # # Extract three-dimensional periodic boundary node coordinates
        # pb_x = []
        # pb_y = []
        # pb_z = []
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     pb_x.append(pb_node_0_x)
        #     pb_x.append(pb_node_1_x)
        #     pb_y.append(pb_node_0_y)
        #     pb_y.append(pb_node_1_y)
        #     pb_z.append(pb_node_0_z)
        #     pb_z.append(pb_node_1_z)
        # pb_x = np.asarray(pb_x)
        # pb_y = np.asarray(pb_y)
        # pb_z = np.asarray(pb_z)

        # # Plot core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(core_x, core_y, core_z, marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_node_coords_filename)
        # plt.close()

        # # Plot core and periodic boundary nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(core_x, core_y, core_z, marker=".", color="black")
        # ax.scatter(pb_x, pb_y, pb_z, marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_node_coords_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the spatial topology of the
        # # core and periodic boundary nodes and edges
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     edge_x = np.asarray(
        #         [
        #             core_node_0_x,
        #             pb_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_node_0_y,
        #             pb_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_node_0_z,
        #             pb_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        #     edge_x = np.asarray(
        #         [
        #             pb_node_0_x,
        #             core_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             pb_node_0_y,
        #             core_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             pb_node_0_z,
        #             core_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_graph_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the spatial topology of the
        # # core and periodic boundary nodes and edges. Here, core edges
        # # are distinguished by purple lines, and periodic boundary edges
        # # are distinguished by olive lines.
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     edge_x = np.asarray(
        #         [
        #             core_node_0_x,
        #             pb_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_node_0_y,
        #             pb_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_node_0_z,
        #             pb_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        #     edge_x = np.asarray(
        #         [
        #             pb_node_0_x,
        #             core_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             pb_node_0_y,
        #             core_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             pb_node_0_z,
        #             core_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the periodic connections
        # # between the core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_pb_edges[edge, 0]],
        #             core_x[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_pb_edges[edge, 0]],
        #             core_y[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_pb_edges[edge, 0]],
        #             core_z[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(conn_graph_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the periodic connections
        # # between the core nodes. Here, core edges are distinguished by
        # # purple lines, and periodic boundary edges are distinguished by
        # # olive lines.
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_pb_edges[edge, 0]],
        #             core_x[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_pb_edges[edge, 0]],
        #             core_y[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_pb_edges[edge, 0]],
        #             core_z[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(conn_graph_colored_topology_synthesis_filename)
        # plt.close()

        # # Tessellated Delaunay triangulation
        # dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        # for tsslltn in range(dim_3_tsslltn_num):
        #     x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        #     y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        #     z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        #     if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        #     else:
        #         core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
        #             L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
        #         tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
        #         tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
        #         tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))
        
        # del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

        # tsslltd_core = (
        #     np.column_stack((tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))
        # )

        # tsslltd_core_voronoi = Voronoi(tsslltd_core)

        # del tsslltd_core

        # simplices = tsslltd_core_voronoi.simplices

        # simplices_edges = np.empty([len(simplices)*6, 2], dtype=int)
        # simplex_edge_indx = 0
        # for simplex in simplices:
        #     node_0 = int(simplex[0])
        #     node_1 = int(simplex[1])
        #     node_2 = int(simplex[2])
        #     node_3 = int(simplex[3])
            
        #     simplices_edges[simplex_edge_indx, 0] = node_0
        #     simplices_edges[simplex_edge_indx, 1] = node_1
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_1
        #     simplices_edges[simplex_edge_indx, 1] = node_2
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_2
        #     simplices_edges[simplex_edge_indx, 1] = node_0
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_0
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_1
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_2
        #     simplex_edge_indx += 1
        
        # simplices_edges = np.unique(np.sort(simplices_edges, axis=1), axis=0)
        # simplices_m = np.shape(simplices_edges)[0]

        # simplices_x = np.asarray([])
        # simplices_y = np.asarray([])
        # simplices_z = np.asarray([])

        # for edge in range(simplices_m):
        #     simplices_x = np.append(
        #         simplices_x, [tsslltd_core_x[simplices_edges[edge, 0]], tsslltd_core_x[simplices_edges[edge, 1]], np.nan])
        #     simplices_y = np.append(
        #         simplices_y, [tsslltd_core_y[simplices_edges[edge, 0]], tsslltd_core_y[simplices_edges[edge, 1]], np.nan])
        #     simplices_z = np.append(
        #         simplices_z, [tsslltd_core_z[simplices_edges[edge, 0]], tsslltd_core_z[simplices_edges[edge, 1]], np.nan])

        # # Plot tessellated core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
        #     tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_node_coords_filename)
        # plt.close()

        # # Plot Delaunay-triangulated simplices connecting the
        # # tessellated core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.plot3D(
        #     simplices_x, simplices_y, simplices_z,
        #     color="tab:blue", linewidth=0.75)
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
        #     tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_voronoi_filename)
        # plt.close()

        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.plot3D(
        #     simplices_x, simplices_y, simplices_z,
        #     color="tab:blue", linewidth=0.75)
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_voronoi_zoomed_in_filename)
        # plt.close()
        pass

def run_voronoi_network_topology_synthesis_plotter(args):
    voronoi_network_topology_synthesis_plotter(*args)

def voronoi_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        config: int,
        params_arr: np.ndarray) -> None:
    # Identification of the sample value for the desired network
    sample = int(np.where((params_arr == (dim, b, n, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Fundamental graph constituents filenames
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Plots filenames
    core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_topology_synthesis" + ".png"
    )
    core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_colored_topology_synthesis" + ".png"
    )
    conn_graph_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_topology_synthesis" + ".png"
    )
    conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_colored_topology_synthesis"
        + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot preformatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L

    min_core = -plt_pad
    max_core = L + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    core_xlim = np.asarray([min_core, max_core])
    core_ylim = np.asarray([min_core, max_core])
    core_zlim = np.asarray([min_core, max_core])

    core_xticks = np.linspace(min_core, max_core, core_tick_steps)
    core_yticks = np.linspace(min_core, max_core, core_tick_steps)
    core_zticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and plot preformatting
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]
        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_2_core_pb_edge_identification
        )

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        # # Import three-dimension specific functions
        # from heterogeneous_spatial_networks_funcs import (
        #     dim_3_tessellation,
        #     dim_3_core_pb_edge_identification
        # )
        
        # # Load fundamental z-dimensional graph constituents
        # core_z = np.loadtxt(core_z_filename)
        # # Tessellated core node z-coordinates
        # tsslltd_core_z = core_z.copy()

        # # Extract three-dimensional periodic boundary node coordinates
        # pb_x = []
        # pb_y = []
        # pb_z = []
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     pb_x.append(pb_node_0_x)
        #     pb_x.append(pb_node_1_x)
        #     pb_y.append(pb_node_0_y)
        #     pb_y.append(pb_node_1_y)
        #     pb_z.append(pb_node_0_z)
        #     pb_z.append(pb_node_1_z)
        # pb_x = np.asarray(pb_x)
        # pb_y = np.asarray(pb_y)
        # pb_z = np.asarray(pb_z)

        # # Plot core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(core_x, core_y, core_z, marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_node_coords_filename)
        # plt.close()

        # # Plot core and periodic boundary nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(core_x, core_y, core_z, marker=".", color="black")
        # ax.scatter(pb_x, pb_y, pb_z, marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_node_coords_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the spatial topology of the
        # # core and periodic boundary nodes and edges
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     edge_x = np.asarray(
        #         [
        #             core_node_0_x,
        #             pb_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_node_0_y,
        #             pb_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_node_0_z,
        #             pb_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        #     edge_x = np.asarray(
        #         [
        #             pb_node_0_x,
        #             core_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             pb_node_0_y,
        #             core_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             pb_node_0_z,
        #             core_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_graph_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the spatial topology of the
        # # core and periodic boundary nodes and edges. Here, core edges
        # # are distinguished by purple lines, and periodic boundary edges
        # # are distinguished by olive lines.
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     core_node_0_x = core_x[conn_pb_edges[edge, 0]]
        #     core_node_0_y = core_y[conn_pb_edges[edge, 0]]
        #     core_node_0_z = core_z[conn_pb_edges[edge, 0]]
        #     core_node_1_x = core_x[conn_pb_edges[edge, 1]]
        #     core_node_1_y = core_y[conn_pb_edges[edge, 1]]
        #     core_node_1_z = core_z[conn_pb_edges[edge, 1]]
        #     pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_1_x, core_node_1_y, core_node_1_z,
        #             core_node_0_x, core_node_0_y, core_node_0_z, L)
        #     )
        #     pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
        #         dim_3_core_pb_edge_identification(
        #             core_node_0_x, core_node_0_y, core_node_0_z,
        #             core_node_1_x, core_node_1_y, core_node_1_z, L)
        #     )
        #     edge_x = np.asarray(
        #         [
        #             core_node_0_x,
        #             pb_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_node_0_y,
        #             pb_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_node_0_z,
        #             pb_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        #     edge_x = np.asarray(
        #         [
        #             pb_node_0_x,
        #             core_node_1_x
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             pb_node_0_y,
        #             core_node_1_y
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             pb_node_0_z,
        #             core_node_1_z
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the periodic connections
        # # between the core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_pb_edges[edge, 0]],
        #             core_x[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_pb_edges[edge, 0]],
        #             core_y[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_pb_edges[edge, 0]],
        #             core_z[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(conn_graph_topology_synthesis_filename)
        # plt.close()

        # # Plot of the unpruned core and periodic boundary cross-linkers
        # # and edges for the graph capturing the periodic connections
        # # between the core nodes. Here, core edges are distinguished by
        # # purple lines, and periodic boundary edges are distinguished by
        # # olive lines.
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # for edge in range(core_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_core_edges[edge, 0]],
        #             core_x[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_core_edges[edge, 0]],
        #             core_y[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_core_edges[edge, 0]],
        #             core_z[conn_core_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # for edge in range(pb_m):
        #     edge_x = np.asarray(
        #         [
        #             core_x[conn_pb_edges[edge, 0]],
        #             core_x[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_y = np.asarray(
        #         [
        #             core_y[conn_pb_edges[edge, 0]],
        #             core_y[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     edge_z = np.asarray(
        #         [
        #             core_z[conn_pb_edges[edge, 0]],
        #             core_z[conn_pb_edges[edge, 1]]
        #         ]
        #     )
        #     ax.plot(
        #         edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
        #         marker=".", markerfacecolor="black", markeredgecolor="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
        #     grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(conn_graph_colored_topology_synthesis_filename)
        # plt.close()

        # # Tessellated Delaunay triangulation
        # dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        # for tsslltn in range(dim_3_tsslltn_num):
        #     x_tsslltn = dim_3_tsslltn[tsslltn, 0]
        #     y_tsslltn = dim_3_tsslltn[tsslltn, 1]
        #     z_tsslltn = dim_3_tsslltn[tsslltn, 2]
        #     if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
        #     else:
        #         core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = dim_3_tessellation(
        #             L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
        #         tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
        #         tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
        #         tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))
        
        # del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

        # tsslltd_core = (
        #     np.column_stack((tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))
        # )

        # tsslltd_core_voronoi = Voronoi(tsslltd_core)

        # del tsslltd_core

        # simplices = tsslltd_core_voronoi.simplices

        # simplices_edges = np.empty([len(simplices)*6, 2], dtype=int)
        # simplex_edge_indx = 0
        # for simplex in simplices:
        #     node_0 = int(simplex[0])
        #     node_1 = int(simplex[1])
        #     node_2 = int(simplex[2])
        #     node_3 = int(simplex[3])
            
        #     simplices_edges[simplex_edge_indx, 0] = node_0
        #     simplices_edges[simplex_edge_indx, 1] = node_1
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_1
        #     simplices_edges[simplex_edge_indx, 1] = node_2
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_2
        #     simplices_edges[simplex_edge_indx, 1] = node_0
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_0
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_1
        #     simplex_edge_indx += 1
        #     simplices_edges[simplex_edge_indx, 0] = node_3
        #     simplices_edges[simplex_edge_indx, 1] = node_2
        #     simplex_edge_indx += 1
        
        # simplices_edges = np.unique(np.sort(simplices_edges, axis=1), axis=0)
        # simplices_m = np.shape(simplices_edges)[0]

        # simplices_x = np.asarray([])
        # simplices_y = np.asarray([])
        # simplices_z = np.asarray([])

        # for edge in range(simplices_m):
        #     simplices_x = np.append(
        #         simplices_x, [tsslltd_core_x[simplices_edges[edge, 0]], tsslltd_core_x[simplices_edges[edge, 1]], np.nan])
        #     simplices_y = np.append(
        #         simplices_y, [tsslltd_core_y[simplices_edges[edge, 0]], tsslltd_core_y[simplices_edges[edge, 1]], np.nan])
        #     simplices_z = np.append(
        #         simplices_z, [tsslltd_core_z[simplices_edges[edge, 0]], tsslltd_core_z[simplices_edges[edge, 1]], np.nan])

        # # Plot tessellated core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
        #     tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_node_coords_filename)
        # plt.close()

        # # Plot Delaunay-triangulated simplices connecting the
        # # tessellated core nodes
        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.plot3D(
        #     simplices_x, simplices_y, simplices_z,
        #     color="tab:blue", linewidth=0.75)
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
        #     tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_voronoi_filename)
        # plt.close()

        # fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        # ax.plot3D(
        #     simplices_x, simplices_y, simplices_z,
        #     color="tab:blue", linewidth=0.75)
        # ax.scatter(
        #     tsslltd_core_x, tsslltd_core_y, tsslltd_core_z,
        #     marker=".", color="black")
        # ax = dim_3_network_topology_axes_formatter(
        #     ax, core_cube, core_cube_color, core_cube_linewidth,
        #     core_xlim, core_ylim, core_zlim,
        #     core_xticks, core_yticks, core_zticks,
        #     xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        # fig.tight_layout()
        # fig.savefig(tsslltd_core_voronoi_zoomed_in_filename)
        # plt.close()
        pass

def run_voronoi_network_topology_plotter(args):
    voronoi_network_topology_plotter(*args)

def delaunay_network_topology_synthesis_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        tsslltd_core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        config: int,
        params_arr: np.ndarray) -> None:
    # Import functions
    from scipy.spatial import Delaunay
    from heterogeneous_spatial_networks_funcs import tessellation_protocol

    # Identification of the sample value for the desired network
    sample = int(
        np.where((params_arr == (dim, b, n, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Fundamental graph constituents filenames
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Plots filenames
    core_node_coords_filename = filename_prefix + "-core_node_coords" + ".png"
    core_pb_node_coords_filename = (
        filename_prefix + "-core_pb_node_coords" + ".png"
    )
    core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_topology_synthesis" + ".png"
    )
    core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_colored_topology_synthesis" + ".png"
    )
    conn_graph_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_topology_synthesis" + ".png"
    )
    conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_colored_topology_synthesis"
        + ".png"
    )
    tsslltd_core_node_coords_filename = (
        filename_prefix + "-tsslltd_core_node_coords" + ".png"
    )
    tsslltd_core_deltri_filename = (
        filename_prefix + "-tsslltd_core_deltri" + ".png"
    )
    tsslltd_core_deltri_zoomed_in_filename = (
        filename_prefix + "-tsslltd_core_deltri_zoomed_in" + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])
    # Tessellated core node coordinates
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = np.asarray([])

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot preformatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    tsslltd_core_tick_inc = tsslltd_core_tick_inc_prefactor * L

    min_core = -plt_pad
    max_core = L + plt_pad
    min_tsslltd_core = -L - plt_pad
    max_tsslltd_core = 2 * L + plt_pad
    
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1
    tsslltd_core_tick_steps = (
        int(np.around((max_tsslltd_core-min_tsslltd_core)/tsslltd_core_tick_inc))
        + 1
    )

    core_xlim = np.asarray([min_core, max_core])
    core_ylim = np.asarray([min_core, max_core])
    core_zlim = np.asarray([min_core, max_core])
    tsslltd_core_xlim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_ylim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_zlim = np.asarray([min_tsslltd_core, max_tsslltd_core])

    core_xticks = np.linspace(min_core, max_core, core_tick_steps)
    core_yticks = np.linspace(min_core, max_core, core_tick_steps)
    core_zticks = np.linspace(min_core, max_core, core_tick_steps)
    tsslltd_core_xticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_yticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_zticks = np.linspace(
            min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and plot preformatting
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]
        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_2_tessellation,
            dim_2_core_pb_edge_identification
        )
        
        # Extract two-dimensional periodic boundary node coordinates
        pb_x = []
        pb_y = []
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            pb_x.append(pb_node_0_x)
            pb_x.append(pb_node_1_x)
            pb_y.append(pb_node_0_y)
            pb_y.append(pb_node_1_y)
        pb_x = np.asarray(pb_x)
        pb_y = np.asarray(pb_y)
        pb_n = np.shape(pb_x)[0]

        # Plot core nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_node_coords_filename)
        plt.close()
        
        # Plot core and periodic boundary nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax.scatter(pb_x, pb_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_node_coords_filename)
        plt.close()

        # Tessellated Delaunay triangulation
        dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
        for tsslltn in range(dim_2_tsslltn_num):
            x_tsslltn = dim_2_tsslltn[tsslltn, 0]
            y_tsslltn = dim_2_tsslltn[tsslltn, 1]
            if (x_tsslltn == 0) and (y_tsslltn == 0): continue
            else:
                core_tsslltn_x, core_tsslltn_y = (
                    dim_2_tessellation(L, core_x, core_y, x_tsslltn, y_tsslltn)
                )
                tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
                tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
        
        del core_tsslltn_x, core_tsslltn_y

        tsslltd_core = np.column_stack((tsslltd_core_x, tsslltd_core_y))

        tsslltd_core_deltri = Delaunay(tsslltd_core)

        del tsslltd_core

        simplices = tsslltd_core_deltri.simplices

        # Plot tessellated core nodes
        fig, ax = plt.subplots()
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_node_coords_filename)
        plt.close()

        # Plot Delaunay-triangulated simplices connecting the
        # tessellated core nodes
        fig, ax = plt.subplots()
        ax.triplot(tsslltd_core_x, tsslltd_core_y, simplices, linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_filename)
        plt.close()

        fig, ax = plt.subplots()
        ax.triplot(tsslltd_core_x, tsslltd_core_y, simplices, linewidth=1.5)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_zoomed_in_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        # Import three-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_3_tessellation,
            dim_3_core_pb_edge_identification
        )
        
        # Load fundamental z-dimensional graph constituents
        core_z = np.loadtxt(core_z_filename)
        # Tessellated core node z-coordinates
        tsslltd_core_z = core_z.copy()

        # Extract three-dimensional periodic boundary node coordinates
        pb_x = []
        pb_y = []
        pb_z = []
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            pb_x.append(pb_node_0_x)
            pb_x.append(pb_node_1_x)
            pb_y.append(pb_node_0_y)
            pb_y.append(pb_node_1_y)
            pb_z.append(pb_node_0_z)
            pb_z.append(pb_node_1_z)
        pb_x = np.asarray(pb_x)
        pb_y = np.asarray(pb_y)
        pb_z = np.asarray(pb_z)

        # Plot core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(core_x, core_y, core_z, s=1.5, marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_node_coords_filename)
        plt.close()

        # Plot core and periodic boundary nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(core_x, core_y, core_z, s=1.5, marker=".", color="black")
        ax.scatter(pb_x, pb_y, pb_z, s=1.5, marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_node_coords_filename)
        plt.close()

        # Tessellated Delaunay triangulation
        dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        for tsslltn in range(dim_3_tsslltn_num):
            x_tsslltn = dim_3_tsslltn[tsslltn, 0]
            y_tsslltn = dim_3_tsslltn[tsslltn, 1]
            z_tsslltn = dim_3_tsslltn[tsslltn, 2]
            if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
            else:
                core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = (
                    dim_3_tessellation(
                        L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
                )
                tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
                tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
                tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))
        
        del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

        tsslltd_core = (
            np.column_stack((tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))
        )

        tsslltd_core_deltri = Delaunay(tsslltd_core)

        del tsslltd_core

        simplices = tsslltd_core_deltri.simplices

        simplices_edges = np.empty([len(simplices)*6, 2], dtype=int)
        simplex_edge_indx = 0
        for simplex in simplices:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])
            node_3 = int(simplex[3])
            
            simplices_edges[simplex_edge_indx, 0] = node_0
            simplices_edges[simplex_edge_indx, 1] = node_1
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_1
            simplices_edges[simplex_edge_indx, 1] = node_2
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_2
            simplices_edges[simplex_edge_indx, 1] = node_0
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_0
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_1
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_2
            simplex_edge_indx += 1
        
        simplices_edges = np.unique(np.sort(simplices_edges, axis=1), axis=0)
        simplices_m = np.shape(simplices_edges)[0]

        simplices_x = np.asarray([])
        simplices_y = np.asarray([])
        simplices_z = np.asarray([])

        for edge in range(simplices_m):
            simplices_x = np.append(
                simplices_x, [tsslltd_core_x[simplices_edges[edge, 0]], tsslltd_core_x[simplices_edges[edge, 1]], np.nan])
            simplices_y = np.append(
                simplices_y, [tsslltd_core_y[simplices_edges[edge, 0]], tsslltd_core_y[simplices_edges[edge, 1]], np.nan])
            simplices_z = np.append(
                simplices_z, [tsslltd_core_z[simplices_edges[edge, 0]], tsslltd_core_z[simplices_edges[edge, 1]], np.nan])

        # Plot tessellated core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
            tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_node_coords_filename)
        plt.close()

        # Plot Delaunay-triangulated simplices connecting the
        # tessellated core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.plot3D(
            simplices_x, simplices_y, simplices_z,
            color="tab:blue", linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
            tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_filename)
        plt.close()

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.plot3D(
            simplices_x, simplices_y, simplices_z,
            color="tab:blue", linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_zoomed_in_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()

def run_delaunay_network_topology_synthesis_plotter(args):
    delaunay_network_topology_synthesis_plotter(*args)

def delaunay_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        config: int,
        params_arr: np.ndarray) -> None:
    # Identification of the sample value for the desired network
    sample = int(
        np.where((params_arr == (dim, b, n, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Fundamental graph constituents filenames
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Plots filenames
    core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_topology_synthesis" + ".png"
    )
    core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_colored_topology_synthesis" + ".png"
    )
    conn_graph_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_topology_synthesis" + ".png"
    )
    conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_colored_topology_synthesis"
        + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot preformatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L

    min_core = -plt_pad
    max_core = L + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    core_xlim = np.asarray([min_core, max_core])
    core_ylim = np.asarray([min_core, max_core])
    core_zlim = np.asarray([min_core, max_core])

    core_xticks = np.linspace(min_core, max_core, core_tick_steps)
    core_yticks = np.linspace(min_core, max_core, core_tick_steps)
    core_zticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and plot preformatting
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]
        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_2_core_pb_edge_identification
        )
        
        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        # Import three-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_3_core_pb_edge_identification
        )
        
        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()

def run_delaunay_network_topology_plotter(args):
    delaunay_network_topology_plotter(*args)

def swidt_network_topology_synthesis_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        tsslltd_core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        k: int,
        eta_n: float,
        config: int,
        params_arr: np.ndarray) -> None:
    # Import functions
    from scipy.spatial import Delaunay
    import networkx as nx
    from heterogeneous_spatial_networks_funcs import (
        tessellation_protocol,
        add_nodes_from_numpy_array,
        add_edges_from_numpy_array
    )

    # Identification of the sample value for the desired network
    sample = int(
        np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Fundamental graph constituents filenames
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Plots filenames
    core_node_coords_filename = filename_prefix + "-core_node_coords" + ".png"
    core_pb_node_coords_filename = (
        filename_prefix + "-core_pb_node_coords" + ".png"
    )
    core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_topology_synthesis" + ".png"
    )
    core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-core_pb_graph_colored_topology_synthesis" + ".png"
    )
    conn_graph_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_topology_synthesis" + ".png"
    )
    conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-conn_graph_colored_topology_synthesis"
        + ".png"
    )
    tsslltd_core_node_coords_filename = (
        filename_prefix + "-tsslltd_core_node_coords" + ".png"
    )
    tsslltd_core_deltri_filename = (
        filename_prefix + "-tsslltd_core_deltri" + ".png"
    )
    tsslltd_core_deltri_zoomed_in_filename = (
        filename_prefix + "-tsslltd_core_deltri_zoomed_in" + ".png"
    )
    pruned_core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-pruned_core_pb_graph_topology_synthesis" + ".png"
    )
    pruned_conn_graph_topology_synthesis_filename = (
        filename_prefix + "-pruned_conn_graph_topology_synthesis" + ".png"
    )
    mx_cmp_pruned_core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-mx_cmp_pruned_core_pb_graph_topology_synthesis" + ".png"
    )
    mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-mx_cmp_pruned_core_pb_graph_colored_topology_synthesis" + ".png"
    )
    mx_cmp_pruned_conn_graph_topology_synthesis_filename = (
        filename_prefix + "-mx_cmp_pruned_conn_graph_topology_synthesis" + ".png"
    )
    mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-mx_cmp_pruned_conn_graph_colored_topology_synthesis" + ".png"
    )
    ee_mx_cmp_pruned_core_pb_graph_topology_synthesis_filename = (
        filename_prefix + "-ee_mx_cmp_pruned_core_pb_graph_topology_synthesis" + ".png"
    )
    ee_mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-ee_mx_cmp_pruned_core_pb_graph_colored_topology_synthesis" + ".png"
    )
    ee_mx_cmp_pruned_conn_graph_topology_synthesis_filename = (
        filename_prefix + "-ee_mx_cmp_pruned_conn_graph_topology_synthesis" + ".png"
    )
    ee_mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename = (
        filename_prefix + "-ee_mx_cmp_pruned_conn_graph_colored_topology_synthesis" + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    core_nodes = np.arange(n, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])
    # Tessellated core node coordinates
    tsslltd_core_x = core_x.copy()
    tsslltd_core_y = core_y.copy()
    tsslltd_core_z = np.asarray([])

    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot preformatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    tsslltd_core_tick_inc = tsslltd_core_tick_inc_prefactor * L

    min_core = -plt_pad
    max_core = L + plt_pad
    min_tsslltd_core = -L - plt_pad
    max_tsslltd_core = 2 * L + plt_pad
    
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1
    tsslltd_core_tick_steps = (
        int(np.around((max_tsslltd_core-min_tsslltd_core)/tsslltd_core_tick_inc))
        + 1
    )

    core_xlim = np.asarray([min_core, max_core])
    core_ylim = np.asarray([min_core, max_core])
    core_zlim = np.asarray([min_core, max_core])
    tsslltd_core_xlim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_ylim = np.asarray([min_tsslltd_core, max_tsslltd_core])
    tsslltd_core_zlim = np.asarray([min_tsslltd_core, max_tsslltd_core])

    core_xticks = np.linspace(min_core, max_core, core_tick_steps)
    core_yticks = np.linspace(min_core, max_core, core_tick_steps)
    core_zticks = np.linspace(min_core, max_core, core_tick_steps)
    tsslltd_core_xticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_yticks = np.linspace(
        min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)
    tsslltd_core_zticks = np.linspace(
            min_tsslltd_core, max_tsslltd_core, tsslltd_core_tick_steps)

    xlabel = "x"
    ylabel = "y"
    zlabel = "z"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and plot preformatting
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]
        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_2_tessellation,
            dim_2_core_pb_edge_identification
        )
        
        # Extract two-dimensional periodic boundary node coordinates
        pb_x = []
        pb_y = []
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            pb_x.append(pb_node_0_x)
            pb_x.append(pb_node_1_x)
            pb_y.append(pb_node_0_y)
            pb_y.append(pb_node_1_y)
        pb_x = np.asarray(pb_x)
        pb_y = np.asarray(pb_y)
        pb_n = np.shape(pb_x)[0]
        
        # # Plot preformatting for non-intersecting circles
        # from matplotlib.collections import PatchCollection

        # min_core_circle_left_x = -plt_pad * 0.5
        # max_core_circle_left_x = plt_pad * 1.5
        # min_core_circle_left_y = (L*0.5) - plt_pad
        # max_core_circle_left_y = (L*0.5) + plt_pad

        # min_core_circle_right_x = L - (plt_pad*1.5)
        # max_core_circle_right_x = L + (plt_pad*0.5)
        # min_core_circle_right_y = (L*0.5) - plt_pad
        # max_core_circle_right_y = (L*0.5) + plt_pad

        # core_circle_left_xlim = (
        #     np.asarray([min_core_circle_left_x, max_core_circle_left_x])
        # )
        # core_circle_left_ylim = (
        #     np.asarray([min_core_circle_left_y, max_core_circle_left_y])
        # )

        # core_circle_right_xlim = (
        #     np.asarray([min_core_circle_right_x, max_core_circle_right_x])
        # )
        # core_circle_right_ylim = (
        #     np.asarray([min_core_circle_right_y, max_core_circle_right_y])
        # )

        # core_circle_num_ticks = 7

        # core_circle_left_xticks = (
        #     np.linspace(
        #         min_core_circle_left_x,
        #         max_core_circle_left_x,
        #         core_circle_num_ticks)
        # )
        # core_circle_left_yticks = (
        #     np.linspace(
        #         min_core_circle_left_y,
        #         max_core_circle_left_y,
        #         core_circle_num_ticks)
        # )
        
        # core_circle_right_xticks = (
        #     np.linspace(
        #         min_core_circle_right_x,
        #         max_core_circle_right_x,
        #         core_circle_num_ticks)
        # )
        # core_circle_right_yticks = (
        #     np.linspace(
        #         min_core_circle_right_y,
        #         max_core_circle_right_y,
        #         core_circle_num_ticks)
        # )

        # min_core_pb_circle_left_x = -plt_pad
        # max_core_pb_circle_left_x = plt_pad
        # min_core_pb_circle_left_y = (L*0.5) - plt_pad
        # max_core_pb_circle_left_y = (L*0.5) + plt_pad

        # min_core_pb_circle_right_x = L - plt_pad
        # max_core_pb_circle_right_x = L + plt_pad
        # min_core_pb_circle_right_y = (L*0.5) - plt_pad
        # max_core_pb_circle_right_y = (L*0.5) + plt_pad

        # core_pb_circle_left_xlim = (
        #     np.asarray([min_core_pb_circle_left_x, max_core_pb_circle_left_x])
        # )
        # core_pb_circle_left_ylim = (
        #     np.asarray([min_core_pb_circle_left_y, max_core_pb_circle_left_y])
        # )

        # core_pb_circle_right_xlim = (
        #     np.asarray([min_core_pb_circle_right_x, max_core_pb_circle_right_x])
        # )
        # core_pb_circle_right_ylim = (
        #     np.asarray([min_core_pb_circle_right_y, max_core_pb_circle_right_y])
        # )

        # core_pb_circle_num_ticks = 7

        # core_pb_circle_left_xticks = (
        #     np.linspace(
        #         min_core_pb_circle_left_x,
        #         max_core_pb_circle_left_x,
        #         core_pb_circle_num_ticks)
        # )
        # core_pb_circle_left_yticks = (
        #     np.linspace(
        #         min_core_pb_circle_left_y,
        #         max_core_pb_circle_left_y,
        #         core_pb_circle_num_ticks)
        # )
        
        # core_pb_circle_right_xticks = (
        #     np.linspace(
        #         min_core_pb_circle_right_x,
        #         max_core_pb_circle_right_x,
        #         core_pb_circle_num_ticks)
        # )
        # core_pb_circle_right_yticks = (
        #     np.linspace(
        #         min_core_pb_circle_right_y,
        #         max_core_pb_circle_right_y,
        #         core_pb_circle_num_ticks)
        # )

        # # Generate filenames
        # core_node_circles_left_filename = (
        #     filename_prefix + "-core_node_circles_left" + ".png"
        # )
        # core_node_circles_right_filename = (
        #     filename_prefix + "-core_node_circles_right" + ".png"
        # )

        # core_pb_node_circles_left_filename = (
        #     filename_prefix + "-core_pb_node_circles_left" + ".png"
        # )
        # core_pb_node_circles_right_filename = (
        #     filename_prefix + "-core_pb_node_circles_right" + ".png"
        # )

        # Plot core nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_node_coords_filename)
        plt.close()

        # # Plot core nodes with non-intersecting circles of diameter = b,
        # # and show a zoomed in portion of the left boundary
        # fig, ax = plt.subplots()
        # ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        # core_circles_left = list(
        #     plt.Circle((core_x[node], core_y[node]), radius=b/2, fill=False) for node in range(n))
        # collection_core_circles_left = (
        #     PatchCollection(core_circles_left, match_original=True)
        # )
        # ax.add_collection(collection_core_circles_left)
        # ax = dim_2_network_topology_axes_formatter(
        #     ax, core_square, core_square_color, core_square_linewidth,
        #     core_circle_left_xlim, core_circle_left_ylim,
        #     core_circle_left_xticks, core_circle_left_yticks, xlabel, ylabel,
        #     grid_alpha, grid_zorder)
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig(core_node_circles_left_filename)
        # plt.close()

        # # Plot core nodes with non-intersecting circles of diameter = b,
        # # and show a zoomed in portion of the right boundary
        # fig, ax = plt.subplots()
        # ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        # core_circles_right = list(
        #     plt.Circle((core_x[node], core_y[node]), radius=b/2, fill=False) for node in range(n))
        # collection_core_circles_right = (
        #     PatchCollection(core_circles_right, match_original=True)
        # )
        # ax.add_collection(collection_core_circles_right)
        # ax = dim_2_network_topology_axes_formatter(
        #     ax, core_square, core_square_color, core_square_linewidth,
        #     core_circle_right_xlim, core_circle_right_ylim,
        #     core_circle_right_xticks, core_circle_right_yticks, xlabel, ylabel,
        #     grid_alpha, grid_zorder)
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig(core_node_circles_right_filename)
        # plt.close()

        # Plot core and periodic boundary nodes
        fig, ax = plt.subplots()
        ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        ax.scatter(pb_x, pb_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_node_coords_filename)
        plt.close()

        # # Plot core and periodic boundary nodes with non-intersecting
        # # circles of diameter = b, and show a zoomed in portion of the
        # # left boundary
        # fig, ax = plt.subplots()
        # ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        # ax.scatter(pb_x, pb_y, s=1.5, marker=".", color="black")
        # core_circles_left = list(
        #     plt.Circle((core_x[node], core_y[node]), radius=b/2, fill=False) for node in range(n))
        # collection_core_circles_left = (
        #     PatchCollection(core_circles_left, match_original=True)
        # )
        # pb_circles_left = list(
        #     plt.Circle((pb_x[node], pb_y[node]), radius=b/2, fill=False) for node in range(pb_n))
        # collection_pb_circles_left = (
        #     PatchCollection(pb_circles_left, match_original=True)
        # )
        # ax.add_collection(collection_core_circles_left)
        # ax.add_collection(collection_pb_circles_left)
        # ax = dim_2_network_topology_axes_formatter(
        #     ax, core_square, core_square_color, core_square_linewidth,
        #     core_pb_circle_left_xlim, core_pb_circle_left_ylim,
        #     core_pb_circle_left_xticks, core_pb_circle_left_yticks,
        #     xlabel, ylabel, grid_alpha, grid_zorder)
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig(core_pb_node_circles_left_filename)
        # plt.close()

        # # Plot core and periodic boundary nodes with non-intersecting
        # # circles of diameter = b, and show a zoomed in portion of the
        # # right boundary
        # fig, ax = plt.subplots()
        # ax.scatter(core_x, core_y, s=1.5, marker=".", color="black")
        # ax.scatter(pb_x, pb_y, s=1.5, marker=".", color="black")
        # core_circles_right = list(
        #     plt.Circle((core_x[node], core_y[node]), radius=b/2, fill=False) for node in range(n))
        # collection_core_circles_right = (
        #     PatchCollection(core_circles_right, match_original=True)
        # )
        # pb_circles_right = list(
        #     plt.Circle((pb_x[node], pb_y[node]), radius=b/2, fill=False) for node in range(pb_n))
        # collection_pb_circles_right = (
        #     PatchCollection(pb_circles_right, match_original=True)
        # )
        # ax.add_collection(collection_core_circles_right)
        # ax.add_collection(collection_pb_circles_right)
        # ax = dim_2_network_topology_axes_formatter(
        #     ax, core_square, core_square_color, core_square_linewidth,
        #     core_pb_circle_right_xlim, core_pb_circle_right_ylim,
        #     core_pb_circle_right_xticks, core_pb_circle_right_yticks,
        #     xlabel, ylabel, grid_alpha, grid_zorder)
        # ax.set_aspect("equal")
        # fig.tight_layout()
        # fig.savefig(core_pb_node_circles_right_filename)
        # plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges. Here, core edges
        # are distinguished by purple lines, and periodic boundary edges
        # are distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes. Here, core edges are distinguished by
        # purple lines, and periodic boundary edges are distinguished by
        # olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()

        # Tessellated Delaunay triangulation
        dim_2_tsslltn, dim_2_tsslltn_num = tessellation_protocol(2)
        
        for tsslltn in range(dim_2_tsslltn_num):
            x_tsslltn = dim_2_tsslltn[tsslltn, 0]
            y_tsslltn = dim_2_tsslltn[tsslltn, 1]
            if (x_tsslltn == 0) and (y_tsslltn == 0): continue
            else:
                core_tsslltn_x, core_tsslltn_y = (
                    dim_2_tessellation(L, core_x, core_y, x_tsslltn, y_tsslltn)
                )
                tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
                tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
        
        del core_tsslltn_x, core_tsslltn_y

        tsslltd_core = np.column_stack((tsslltd_core_x, tsslltd_core_y))

        tsslltd_core_deltri = Delaunay(tsslltd_core)

        del tsslltd_core

        simplices = tsslltd_core_deltri.simplices

        # Plot tessellated core nodes
        fig, ax = plt.subplots()
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_node_coords_filename)
        plt.close()

        # Plot Delaunay-triangulated simplices connecting the
        # tessellated core nodes
        fig, ax = plt.subplots()
        ax.triplot(tsslltd_core_x, tsslltd_core_y, simplices, linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim,
            tsslltd_core_xticks, tsslltd_core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_filename)
        plt.close()

        fig, ax = plt.subplots()
        ax.triplot(tsslltd_core_x, tsslltd_core_y, simplices, linewidth=1.5)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, s=1.5, marker=".", color="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_zoomed_in_filename)
        plt.close()
    elif dim == 3:
        # Import three-dimension specific functions
        from heterogeneous_spatial_networks_funcs import (
            dim_3_tessellation,
            dim_3_core_pb_edge_identification
        )
        
        # Load fundamental z-dimensional graph constituents
        core_z = np.loadtxt(core_z_filename)
        # Tessellated core node z-coordinates
        tsslltd_core_z = core_z.copy()

        # Extract three-dimensional periodic boundary node coordinates
        pb_x = []
        pb_y = []
        pb_z = []
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            pb_x.append(pb_node_0_x)
            pb_x.append(pb_node_1_x)
            pb_y.append(pb_node_0_y)
            pb_y.append(pb_node_1_y)
            pb_z.append(pb_node_0_z)
            pb_z.append(pb_node_1_z)
        pb_x = np.asarray(pb_x)
        pb_y = np.asarray(pb_y)
        pb_z = np.asarray(pb_z)

        # Plot core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(core_x, core_y, core_z, s=1.5, marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_node_coords_filename)
        plt.close()

        # Plot core and periodic boundary nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(core_x, core_y, core_z, s=1.5, marker=".", color="black")
        ax.scatter(pb_x, pb_y, pb_z, s=1.5, marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_node_coords_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges. Here, core edges
        # are distinguished by purple lines, and periodic boundary edges
        # are distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes. Here, core edges are distinguished by
        # purple lines, and periodic boundary edges are distinguished by
        # olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_synthesis_filename)
        plt.close()

        # Tessellated Delaunay triangulation
        dim_3_tsslltn, dim_3_tsslltn_num = tessellation_protocol(3)
        
        for tsslltn in range(dim_3_tsslltn_num):
            x_tsslltn = dim_3_tsslltn[tsslltn, 0]
            y_tsslltn = dim_3_tsslltn[tsslltn, 1]
            z_tsslltn = dim_3_tsslltn[tsslltn, 2]
            if (x_tsslltn == 0) and (y_tsslltn == 0) and (z_tsslltn == 0): continue
            else:
                core_tsslltn_x, core_tsslltn_y, core_tsslltn_z = (
                    dim_3_tessellation(
                        L, core_x, core_y, core_z, x_tsslltn, y_tsslltn, z_tsslltn)
                )
                tsslltd_core_x = np.concatenate((tsslltd_core_x, core_tsslltn_x))
                tsslltd_core_y = np.concatenate((tsslltd_core_y, core_tsslltn_y))
                tsslltd_core_z = np.concatenate((tsslltd_core_z, core_tsslltn_z))
        
        del core_tsslltn_x, core_tsslltn_y, core_tsslltn_z

        tsslltd_core = (
            np.column_stack((tsslltd_core_x, tsslltd_core_y, tsslltd_core_z))
        )

        tsslltd_core_deltri = Delaunay(tsslltd_core)

        del tsslltd_core

        simplices = tsslltd_core_deltri.simplices

        simplices_edges = np.empty([len(simplices)*6, 2], dtype=int)
        simplex_edge_indx = 0
        for simplex in simplices:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])
            node_3 = int(simplex[3])
            
            simplices_edges[simplex_edge_indx, 0] = node_0
            simplices_edges[simplex_edge_indx, 1] = node_1
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_1
            simplices_edges[simplex_edge_indx, 1] = node_2
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_2
            simplices_edges[simplex_edge_indx, 1] = node_0
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_0
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_1
            simplex_edge_indx += 1
            simplices_edges[simplex_edge_indx, 0] = node_3
            simplices_edges[simplex_edge_indx, 1] = node_2
            simplex_edge_indx += 1
        
        simplices_edges = np.unique(np.sort(simplices_edges, axis=1), axis=0)
        simplices_m = np.shape(simplices_edges)[0]

        simplices_x = np.asarray([])
        simplices_y = np.asarray([])
        simplices_z = np.asarray([])

        for edge in range(simplices_m):
            simplices_x = np.append(
                simplices_x, [tsslltd_core_x[simplices_edges[edge, 0]], tsslltd_core_x[simplices_edges[edge, 1]], np.nan])
            simplices_y = np.append(
                simplices_y, [tsslltd_core_y[simplices_edges[edge, 0]], tsslltd_core_y[simplices_edges[edge, 1]], np.nan])
            simplices_z = np.append(
                simplices_z, [tsslltd_core_z[simplices_edges[edge, 0]], tsslltd_core_z[simplices_edges[edge, 1]], np.nan])

        # Plot tessellated core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
            tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_node_coords_filename)
        plt.close()

        # Plot Delaunay-triangulated simplices connecting the
        # tessellated core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.plot3D(
            simplices_x, simplices_y, simplices_z,
            color="tab:blue", linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            tsslltd_core_xlim, tsslltd_core_ylim, tsslltd_core_zlim,
            tsslltd_core_xticks, tsslltd_core_yticks, tsslltd_core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_filename)
        plt.close()

        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        ax.plot3D(
            simplices_x, simplices_y, simplices_z,
            color="tab:blue", linewidth=0.75)
        ax.scatter(
            tsslltd_core_x, tsslltd_core_y, tsslltd_core_z, s=1.5,
            marker=".", color="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(tsslltd_core_deltri_zoomed_in_filename)
        plt.close()
    
    # Edge pruning procedure
    rng = np.random.default_rng()

    # Initialize node number integer constants
    core_node_0 = 0
    core_node_1 = 0

    conn_core_graph = nx.Graph()
    conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    conn_pb_graph = nx.Graph()
    conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    conn_graph = nx.Graph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    conn_graph_k = np.asarray(list(conn_graph.degree()), dtype=int)[:, 1]

    # Explicit edge pruning procedure
    if np.any(conn_graph_k > k):
        while np.any(conn_graph_k > k):
            conn_graph_hyprconn_nodes = np.where(conn_graph_k > k)[0]
            conn_graph_hyprconn_edge_indcs_0 = (
                np.where(np.isin(conn_edges[:, 0], conn_graph_hyprconn_nodes))[0]
            )
            conn_graph_hyprconn_edge_indcs_1 = (
                np.where(np.isin(conn_edges[:, 1], conn_graph_hyprconn_nodes))[0]
            )
            conn_graph_hyprconn_edge_indcs = (
                np.unique(
                    np.concatenate(
                        (conn_graph_hyprconn_edge_indcs_0, conn_graph_hyprconn_edge_indcs_1),
                        dtype=int))
            )
            edge_indcs_indx2remove_indx = (
                rng.integers(
                    np.shape(conn_graph_hyprconn_edge_indcs)[0], dtype=int)
            )
            edge_indx2remove = (
                conn_graph_hyprconn_edge_indcs[edge_indcs_indx2remove_indx]
            )
            core_node_0 = int(conn_edges[edge_indx2remove, 0])
            core_node_1 = int(conn_edges[edge_indx2remove, 1])

            conn_graph.remove_edge(core_node_0, core_node_1)
            conn_edges = np.delete(conn_edges, edge_indx2remove, axis=0)
            if conn_core_graph.has_edge(core_node_0, core_node_1):
                conn_core_graph.remove_edge(core_node_0, core_node_1)
            elif conn_pb_graph.has_edge(core_node_0, core_node_1):
                conn_pb_graph.remove_edge(core_node_0, core_node_1)

            conn_graph_k[core_node_0] -= 1
            conn_graph_k[core_node_1] -= 1
    
    pruned_conn_core_edges = np.asarray(list(conn_core_graph.edges()), dtype=int)
    pruned_conn_pb_edges = np.asarray(list(conn_pb_graph.edges()), dtype=int)
    pruned_core_m = np.shape(pruned_conn_core_edges)[0]
    pruned_pb_m = np.shape(pruned_conn_pb_edges)[0]
    
    if dim == 2:
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(pruned_core_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_core_edges[edge, 0]],
                    core_x[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_core_edges[edge, 0]],
                    core_y[pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pruned_pb_m):
            core_node_0_x = core_x[pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(pruned_core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots()
        for edge in range(pruned_core_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_core_edges[edge, 0]],
                    core_x[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_core_edges[edge, 0]],
                    core_y[pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pruned_pb_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_pb_edges[edge, 0]],
                    core_x[pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_pb_edges[edge, 0]],
                    core_y[pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(pruned_conn_graph_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(pruned_core_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_core_edges[edge, 0]],
                    core_x[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_core_edges[edge, 0]],
                    core_y[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[pruned_conn_core_edges[edge, 0]],
                    core_z[pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pruned_pb_m):
            core_node_0_x = core_x[pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[pruned_conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[pruned_conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(pruned_core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(pruned_core_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_core_edges[edge, 0]],
                    core_x[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_core_edges[edge, 0]],
                    core_y[pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[pruned_conn_core_edges[edge, 0]],
                    core_z[pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pruned_pb_m):
            edge_x = np.asarray(
                [
                    core_x[pruned_conn_pb_edges[edge, 0]],
                    core_x[pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[pruned_conn_pb_edges[edge, 0]],
                    core_y[pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[pruned_conn_pb_edges[edge, 0]],
                    core_z[pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(pruned_conn_graph_topology_synthesis_filename)
        plt.close()
    
    # Isolate largest/maximum connected component from the graph
    mx_cmp_pruned_conn_graph_nodes = max(
        nx.connected_components(conn_graph), key=len)
    mx_cmp_pruned_conn_graph = (
        conn_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
    )
    mx_cmp_pruned_conn_core_graph = (
        conn_core_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
    )
    mx_cmp_pruned_conn_pb_graph = (
        conn_pb_graph.subgraph(mx_cmp_pruned_conn_graph_nodes).copy()
    )
    mx_cmp_pruned_conn_core_graph_edges = np.asarray(
        list(mx_cmp_pruned_conn_core_graph.edges()), dtype=int)
    mx_cmp_pruned_conn_pb_graph_edges = np.asarray(
        list(mx_cmp_pruned_conn_pb_graph.edges()), dtype=int)
    mx_cmp_pruned_conn_core_graph_m = (
        np.shape(mx_cmp_pruned_conn_core_graph_edges)[0]
    )
    mx_cmp_pruned_conn_pb_graph_m = (
        np.shape(mx_cmp_pruned_conn_pb_graph_edges)[0]
    )
    mx_cmp_pruned_conn_graph_nodes = (
        np.sort(np.fromiter(mx_cmp_pruned_conn_graph_nodes, dtype=int))
    )

    mx_cmp_pruned_core_x = core_x[mx_cmp_pruned_conn_graph_nodes]
    mx_cmp_pruned_core_y = core_y[mx_cmp_pruned_conn_graph_nodes]
    mx_cmp_pruned_core_z = np.asarray([])

    for edge in range(mx_cmp_pruned_conn_core_graph_m):
        mx_cmp_pruned_conn_core_graph_edges[edge, 0] = (
            int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_core_graph_edges[edge, 0])[0][0])
        )
        mx_cmp_pruned_conn_core_graph_edges[edge, 1] = (
            int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_core_graph_edges[edge, 1])[0][0])
        )

    for edge in range(mx_cmp_pruned_conn_pb_graph_m):
        mx_cmp_pruned_conn_pb_graph_edges[edge, 0] = (
            int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_pb_graph_edges[edge, 0])[0][0])
        )
        mx_cmp_pruned_conn_pb_graph_edges[edge, 1] = (
            int(np.where(mx_cmp_pruned_conn_graph_nodes == mx_cmp_pruned_conn_pb_graph_edges[edge, 1])[0][0])
        )
    
    mx_cmp_pruned_core_m = mx_cmp_pruned_conn_core_graph_m
    mx_cmp_pruned_pb_m = mx_cmp_pruned_conn_pb_graph_m
    
    if dim == 2:
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges.
        # Here, core edges are distinguished by purple lines, and
        # periodic boundary edges are distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename)
        plt.close()
    elif dim == 3:
        mx_cmp_pruned_core_z = core_z[mx_cmp_pruned_conn_graph_nodes]

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges.
        # Here, core edges are distinguished by purple lines, and
        # periodic boundary edges are distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_0_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            core_node_1_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_topology_synthesis_filename)
        plt.close()

        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            core_xlim, core_ylim, core_zlim,
            core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
            grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename)
        plt.close()
    
    # # Modified elastically-effective pruning procedure
    # from networkx.algorithms.connectivity.edge_kcomponents import bridge_components

    # # Isolate largest/maximum bridge component in a nodewise fashion
    # ee_mx_cmp_pruned_conn_graph_nodes = max(
    #     bridge_components(mx_cmp_pruned_conn_graph), key=len)
    # # Extract largest/maximum bridge component subgraphs
    # ee_mx_cmp_pruned_conn_core_graph = (
    #     mx_cmp_pruned_conn_core_graph.subgraph(ee_mx_cmp_pruned_conn_graph_nodes).copy()
    # )
    # ee_mx_cmp_pruned_conn_pb_graph = (
    #     mx_cmp_pruned_conn_pb_graph.subgraph(ee_mx_cmp_pruned_conn_graph_nodes).copy()
    # )
    # ee_mx_cmp_pruned_conn_core_graph_edges = np.asarray(
    #     list(ee_mx_cmp_pruned_conn_core_graph.edges()), dtype=int)
    # ee_mx_cmp_pruned_conn_pb_graph_edges = np.asarray(
    #     list(ee_mx_cmp_pruned_conn_pb_graph.edges()), dtype=int)
    # ee_mx_cmp_pruned_conn_core_graph_m = (
    #     np.shape(ee_mx_cmp_pruned_conn_core_graph_edges)[0]
    # )
    # ee_mx_cmp_pruned_conn_pb_graph_m = (
    #     np.shape(ee_mx_cmp_pruned_conn_pb_graph_edges)[0]
    # )
    # ee_mx_cmp_pruned_conn_graph_nodes = (
    #     np.sort(np.fromiter(ee_mx_cmp_pruned_conn_graph_nodes, dtype=int))
    # )

    # ee_mx_cmp_pruned_core_x = core_x[ee_mx_cmp_pruned_conn_graph_nodes]
    # ee_mx_cmp_pruned_core_y = core_y[ee_mx_cmp_pruned_conn_graph_nodes]
    # ee_mx_cmp_pruned_core_z = np.asarray([])

    # for edge in range(ee_mx_cmp_pruned_conn_core_graph_m):
    #     ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0] = (
    #         int(np.where(ee_mx_cmp_pruned_conn_graph_nodes == ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0])[0][0])
    #     )
    #     ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1] = (
    #         int(np.where(ee_mx_cmp_pruned_conn_graph_nodes == ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1])[0][0])
    #     )

    # for edge in range(ee_mx_cmp_pruned_conn_pb_graph_m):
    #     ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0] = (
    #         int(np.where(ee_mx_cmp_pruned_conn_graph_nodes == ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0])[0][0])
    #     )
    #     ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1] = (
    #         int(np.where(ee_mx_cmp_pruned_conn_graph_nodes == ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1])[0][0])
    #     )
    
    # ee_mx_cmp_pruned_core_m = ee_mx_cmp_pruned_conn_core_graph_m
    # ee_mx_cmp_pruned_pb_m = ee_mx_cmp_pruned_conn_pb_graph_m

    # if dim == 2:
    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_1_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges.
    #     # Here, core edges are distinguished by purple lines, and
    #     # periodic boundary edges are distinguished by olive lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_1_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         core_xlim, core_ylim, core_xticks, core_yticks, xlabel, ylabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename)
    #     plt.close()
    # elif dim == 3:
    #     ee_mx_cmp_pruned_core_z = core_z[ee_mx_cmp_pruned_conn_graph_nodes]

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_z = ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_1_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_z = ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         core_xlim, core_ylim, core_zlim,
    #         core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges.
    #     # Here, core edges are distinguished by purple lines, and
    #     # periodic boundary edges are distinguished by olive lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_0_z = ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]]
    #         core_node_1_x = ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_y = ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         core_node_1_z = ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         core_xlim, core_ylim, core_zlim,
    #         core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_colored_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         core_xlim, core_ylim, core_zlim,
    #         core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_topology_synthesis_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 0]],
    #                 ee_mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_graph_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         core_xlim, core_ylim, core_zlim,
    #         core_xticks, core_yticks, core_zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_colored_topology_synthesis_filename)
    #     plt.close()

def run_swidt_network_topology_synthesis_plotter(args):
    swidt_network_topology_synthesis_plotter(*args)

def swidt_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n: int,
        k: int,
        eta_n: float,
        config: int,
        pruning: int,
        params_arr: np.ndarray) -> None:
    # Import functions
    import networkx as nx
    from heterogeneous_spatial_networks_funcs import (
        add_nodes_from_numpy_array,
        add_edges_from_numpy_array
    )
    
    # Identification of the sample value for the desired network
    sample = int(
        np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Filenames for fundamental graph constituents for unpruned topology
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Filenames for plots for unpruned topology
    core_pb_graph_topology_filename = (
        filename_prefix + "-core_pb_graph_topology" + ".png"
    )
    core_pb_graph_colored_topology_filename = (
        filename_prefix + "-core_pb_graph_colored_topology" + ".png"
    )
    conn_graph_topology_filename = (
        filename_prefix + "-conn_graph_topology" + ".png"
    )
    conn_graph_colored_topology_filename = (
        filename_prefix + "-conn_graph_colored_topology" + ".png"
    )
    # Filenames for fundamental graph constituents for pruned topology
    filename_prefix = filename_prefix + f"P{pruning:d}"
    mx_cmp_pruned_conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    mx_cmp_pruned_conn_core_edges_filename = (
        filename_prefix + "-conn_core_edges" + ".dat"
    )
    mx_cmp_pruned_conn_pb_edges_filename = (
        filename_prefix + "-conn_pb_edges" + ".dat"
    )
    mx_cmp_pruned_core_x_filename = filename_prefix + "-core_x" + ".dat"
    mx_cmp_pruned_core_y_filename = filename_prefix + "-core_y" + ".dat"
    mx_cmp_pruned_core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Filenames for plots for pruned topology
    mx_cmp_pruned_core_pb_graph_topology_filename = (
        filename_prefix + "-core_pb_graph_topology" + ".png"
    )
    mx_cmp_pruned_core_pb_graph_colored_topology_filename = (
        filename_prefix + "-core_pb_graph_colored_topology" + ".png"
    )
    mx_cmp_pruned_conn_graph_topology_filename = (
        filename_prefix + "-conn_graph_topology" + ".png"
    )
    mx_cmp_pruned_conn_graph_colored_topology_filename = (
        filename_prefix + "-conn_graph_colored_topology" + ".png"
    )
    ee_mx_cmp_pruned_core_pb_graph_topology_filename = (
        filename_prefix + "-ee_core_pb_graph_topology" + ".png"
    )
    ee_mx_cmp_pruned_core_pb_graph_colored_topology_filename = (
        filename_prefix + "-ee_core_pb_graph_colored_topology" + ".png"
    )
    ee_mx_cmp_pruned_conn_graph_topology_filename = (
        filename_prefix + "-ee_conn_graph_topology" + ".png"
    )
    ee_mx_cmp_pruned_conn_graph_colored_topology_filename = (
        filename_prefix + "-ee_conn_graph_colored_topology" + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    # Fundamental graph constituents for unpruned topology
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])
    # Fundamental graph constituents for pruned topology
    mx_cmp_pruned_core_nodes = np.arange(
        np.loadtxt(mx_cmp_pruned_conn_n_filename, dtype=int), dtype=int)
    mx_cmp_pruned_conn_core_edges = np.loadtxt(
        mx_cmp_pruned_conn_core_edges_filename, dtype=int)
    mx_cmp_pruned_conn_pb_edges = np.loadtxt(
        mx_cmp_pruned_conn_pb_edges_filename, dtype=int)
    mx_cmp_pruned_core_x = np.loadtxt(mx_cmp_pruned_core_x_filename)
    mx_cmp_pruned_core_y = np.loadtxt(mx_cmp_pruned_core_y_filename)
    mx_cmp_pruned_core_z = np.asarray([])

    # Create nx.Graphs and add nodes before edges
    mx_cmp_pruned_conn_graph = nx.Graph()
    mx_cmp_pruned_conn_graph = add_nodes_from_numpy_array(
        mx_cmp_pruned_conn_graph, mx_cmp_pruned_core_nodes)
    mx_cmp_pruned_conn_graph = add_edges_from_numpy_array(
        mx_cmp_pruned_conn_graph, mx_cmp_pruned_conn_core_edges)
    mx_cmp_pruned_conn_graph = add_edges_from_numpy_array(
        mx_cmp_pruned_conn_graph, mx_cmp_pruned_conn_pb_edges)
    
    mx_cmp_pruned_conn_core_graph = nx.Graph()
    mx_cmp_pruned_conn_core_graph = add_nodes_from_numpy_array(
        mx_cmp_pruned_conn_core_graph, mx_cmp_pruned_core_nodes)
    mx_cmp_pruned_conn_core_graph = add_edges_from_numpy_array(
        mx_cmp_pruned_conn_core_graph, mx_cmp_pruned_conn_core_edges)

    mx_cmp_pruned_conn_pb_graph = nx.Graph()
    mx_cmp_pruned_conn_pb_graph = add_nodes_from_numpy_array(
        mx_cmp_pruned_conn_pb_graph, mx_cmp_pruned_core_nodes)
    mx_cmp_pruned_conn_pb_graph = add_edges_from_numpy_array(
        mx_cmp_pruned_conn_pb_graph, mx_cmp_pruned_conn_pb_edges)
    
    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]
    mx_cmp_pruned_core_m = np.shape(mx_cmp_pruned_conn_core_edges)[0]
    mx_cmp_pruned_pb_m = np.shape(mx_cmp_pruned_conn_pb_edges)[0]

    # Plot formatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    min_core = -plt_pad
    max_core = L + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    xlim = np.asarray([min_core, max_core])
    ylim = np.asarray([min_core, max_core])

    xticks = np.linspace(min_core, max_core, core_tick_steps)
    yticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and preformating
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Plot formatting parameters and preformatting
    zlim = np.asarray([min_core, max_core])
    zticks = np.linspace(min_core, max_core, core_tick_steps)
    zlabel = "z"

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]

        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific function
        from heterogeneous_spatial_networks_funcs import (
            dim_2_core_pb_edge_identification
        )

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_filename)
        plt.close()
        
        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges. Here, core edges
        # are distinguished by purple lines, and periodic boundary edges
        # are distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_filename)
        plt.close()
        
        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes. Here, core edges are distinguished by
        # purple lines, and periodic boundary edges are distinguished by
        # olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges.
        # Here, core edges are distinguished by purple lines, and
        # periodic boundary edges are distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
            pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
                core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_colored_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_colored_topology_filename)
        plt.close()
    elif dim == 3:
        # Import three-dimension specific function
        from heterogeneous_spatial_networks_funcs import (
            dim_3_core_pb_edge_identification
        )
        
        # Load fundamental z-dimensional graph constituents
        core_z = np.loadtxt(core_z_filename)
        mx_cmp_pruned_core_z = np.loadtxt(mx_cmp_pruned_core_z_filename)

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the spatial topology of the
        # core and periodic boundary nodes and edges. Here, core edges
        # are distinguished by purple lines, and periodic boundary edges
        # are distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            core_node_0_x = core_x[conn_pb_edges[edge, 0]]
            core_node_0_y = core_y[conn_pb_edges[edge, 0]]
            core_node_0_z = core_z[conn_pb_edges[edge, 0]]
            core_node_1_x = core_x[conn_pb_edges[edge, 1]]
            core_node_1_y = core_y[conn_pb_edges[edge, 1]]
            core_node_1_z = core_z[conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_filename)
        plt.close()
        
        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_filename)
        plt.close()

        # Plot of the unpruned core and periodic boundary cross-linkers
        # and edges for the graph capturing the periodic connections
        # between the core nodes. Here, core edges are distinguished by
        # purple lines, and periodic boundary edges are distinguished by
        # olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            edge_x = np.asarray(
                [
                    core_x[conn_core_edges[edge, 0]],
                    core_x[conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_core_edges[edge, 0]],
                    core_y[conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_core_edges[edge, 0]],
                    core_z[conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(pb_m):
            edge_x = np.asarray(
                [
                    core_x[conn_pb_edges[edge, 0]],
                    core_x[conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    core_y[conn_pb_edges[edge, 0]],
                    core_y[conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    core_z[conn_pb_edges[edge, 0]],
                    core_z[conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the spatial
        # topology of the core and periodic boundary nodes and edges.
        # Here, core edges are distinguished by purple lines, and
        # periodic boundary edges are distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            core_node_0_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_0_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 0]]
            core_node_1_x = mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_y = mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            core_node_1_z = mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 1]]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_core_pb_graph_colored_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_topology_filename)
        plt.close()
        
        # Plot of the edge pruned core and periodic boundary
        # cross-linkers and edges for the graph capturing the periodic
        # connections between the core nodes. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(mx_cmp_pruned_core_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_core_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        for edge in range(mx_cmp_pruned_pb_m):
            edge_x = np.asarray(
                [
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_x[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_y = np.asarray(
                [
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_y[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            edge_z = np.asarray(
                [
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 0]],
                    mx_cmp_pruned_core_z[mx_cmp_pruned_conn_pb_edges[edge, 1]]
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                marker=".", markersize=1.5, markerfacecolor="black",
                markeredgecolor="black")
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(mx_cmp_pruned_conn_graph_colored_topology_filename)
        plt.close()
    
    # # Modified elastically-effective pruning procedure
    # from networkx.algorithms.connectivity.edge_kcomponents import bridge_components

    # # Isolate largest/maximum bridge component in a nodewise fashion
    # ee_mx_cmp_pruned_conn_graph_nodes = max(
    #     bridge_components(mx_cmp_pruned_conn_graph), key=len)
    # # Extract largest/maximum bridge component subgraphs
    # ee_mx_cmp_pruned_conn_core_graph = (
    #     mx_cmp_pruned_conn_core_graph.subgraph(ee_mx_cmp_pruned_conn_graph_nodes).copy()
    # )
    # ee_mx_cmp_pruned_conn_pb_graph = (
    #     mx_cmp_pruned_conn_pb_graph.subgraph(ee_mx_cmp_pruned_conn_graph_nodes).copy()
    # )
    # ee_mx_cmp_pruned_conn_core_edges = np.asarray(
    #     list(ee_mx_cmp_pruned_conn_core_graph.edges()), dtype=int)
    # ee_mx_cmp_pruned_conn_pb_edges = np.asarray(
    #     list(ee_mx_cmp_pruned_conn_pb_graph.edges()), dtype=int)
    # ee_mx_cmp_pruned_core_m = (
    #     np.shape(ee_mx_cmp_pruned_conn_core_edges)[0]
    # )
    # ee_mx_cmp_pruned_pb_m = (
    #     np.shape(ee_mx_cmp_pruned_conn_pb_edges)[0]
    # )

    # if dim == 2:
    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_1_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges.
    #     # Here, core edges are distinguished by purple lines, and
    #     # periodic boundary edges are distinguished by olive lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_1_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_1_x, core_node_1_y, core_node_0_x, core_node_0_y, L)
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = dim_2_core_pb_edge_identification(
    #             core_node_0_x, core_node_0_y, core_node_1_x, core_node_1_y, L)
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_colored_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_colored_topology_filename)
    #     plt.close()
    # elif dim == 3:
    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_z = mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_1_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_z = mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the spatial
    #     # topology of the core and periodic boundary nodes and edges.
    #     # Here, core edges are distinguished by purple lines, and
    #     # periodic boundary edges are distinguished by olive lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         core_node_0_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_0_z = mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]]
    #         core_node_1_x = mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_y = mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         core_node_1_z = mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )network
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_core_pb_graph_colored_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_topology_filename)
    #     plt.close()

    #     # Plot of the edge pruned core and periodic boundary
    #     # cross-linkers and edges for the graph capturing the periodic
    #     # connections between the core nodes. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_mx_cmp_pruned_core_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_core_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_mx_cmp_pruned_pb_m):
    #         edge_x = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_x[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_y[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 0]],
    #                 mx_cmp_pruned_core_z[ee_mx_cmp_pruned_conn_pb_edges[edge, 1]]
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             marker=".", markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks, xlabel, ylabel, zlabel,
    #         grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_mx_cmp_pruned_conn_graph_colored_topology_filename)
    #     plt.close()

def run_swidt_network_topology_plotter(args):
    swidt_network_topology_plotter(*args)

def aelp_network_core_node_marker_style(
        core_node: int,
        core_node_type: int,
        selfloop_edges: list[tuple[int, int]]) -> tuple[str, str, str]:
    marker = ""
    markerfacecolor = ""
    markeredgecolor = ""
    if core_node_type == 1:
        markerfacecolor = "black"
        markeredgecolor = "black"
        if len(selfloop_edges) == 0: marker = "."
        else:
            core_node_selfloop_edge_order = 0
            for selfloop_edge in selfloop_edges:
                if (selfloop_edge[0] == core_node) and (selfloop_edge[1] == core_node):
                    core_node_selfloop_edge_order += 1
            if core_node_selfloop_edge_order == 0: marker = "."
            elif core_node_selfloop_edge_order == 1: marker = "s"
            elif core_node_selfloop_edge_order == 2: marker = "h"
            elif core_node_selfloop_edge_order >= 3: marker = "8"
    elif core_node_type == 3:
        marker = "."
        markerfacecolor = "red"
        markeredgecolor = "red"
    return (marker, markerfacecolor, markeredgecolor)

def aelp_network_edge_alpha(core_node_0: int, core_node_1: int, conn_graph) -> float:
    alpha = 0.25 * conn_graph.number_of_edges(core_node_0, core_node_1)
    alpha = np.minimum(alpha, 1.0)
    return alpha

def aelp_network_topology_plotter(
        plt_pad_prefactor: float,
        core_tick_inc_prefactor: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        xi: float,
        rho_nu: float,
        k: int,
        n: int,
        nu: int,
        config: int,
        params_arr: np.ndarray) -> None:
    # Import functions
    import networkx as nx
    from heterogeneous_spatial_networks_funcs import (
        add_nodes_from_numpy_array,
        add_edges_from_numpy_array
    )
    
    # Identification of the sample value for the desired network
    
    sample = int(
        np.where((params_arr == (dim, b, xi, rho_nu, k, n, nu)).all(axis=1))[0][0])

    # Generate filenames
    filename_prefix = filename_str(network, date, batch, sample)
    L_filename = filename_prefix + "-L" + ".dat"
    # Filenames for fundamental graph constituents
    filename_prefix = filename_prefix + f"C{config:d}"
    conn_n_filename = filename_prefix + "-conn_n" + ".dat"
    core_node_type_filename = filename_prefix + "-core_node_type" + ".dat"
    conn_core_edges_filename = filename_prefix + "-conn_core_edges" + ".dat"
    conn_pb_edges_filename = filename_prefix + "-conn_pb_edges" + ".dat"
    core_x_filename = filename_prefix + "-core_x" + ".dat"
    core_y_filename = filename_prefix + "-core_y" + ".dat"
    core_z_filename = filename_prefix + "-core_z" + ".dat"
    # Filenames for plots
    core_pb_graph_topology_filename = (
        filename_prefix + "-core_pb_graph_topology" + ".png"
    )
    core_pb_graph_colored_topology_filename = (
        filename_prefix + "-core_pb_graph_colored_topology" + ".png"
    )
    conn_graph_topology_filename = (
        filename_prefix + "-conn_graph_topology" + ".png"
    )
    conn_graph_colored_topology_filename = (
        filename_prefix + "-conn_graph_colored_topology" + ".png"
    )
    ee_core_pb_graph_topology_filename = (
        filename_prefix + "-ee_core_pb_graph_topology" + ".png"
    )
    ee_core_pb_graph_colored_topology_filename = (
        filename_prefix + "-ee_core_pb_graph_colored_topology" + ".png"
    )
    ee_conn_graph_topology_filename = (
        filename_prefix + "-ee_conn_graph_topology" + ".png"
    )
    ee_conn_graph_colored_topology_filename = (
        filename_prefix + "-ee_conn_graph_colored_topology" + ".png"
    )

    # Load fundamental graph constituents
    L = np.loadtxt(L_filename)
    core_nodes = np.arange(np.loadtxt(conn_n_filename, dtype=int), dtype=int)
    core_node_type = np.loadtxt(core_node_type_filename, dtype=int)
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_edges = np.vstack((conn_core_edges, conn_pb_edges), dtype=int)
    core_x = np.loadtxt(core_x_filename)
    core_y = np.loadtxt(core_y_filename)
    core_z = np.asarray([])

    # Create nx.MultiGraph and add nodes before edges
    conn_graph = nx.MultiGraph()
    conn_graph = add_nodes_from_numpy_array(conn_graph, core_nodes)
    conn_graph = add_edges_from_numpy_array(conn_graph, conn_edges)

    # Extract list of self-loop edges
    conn_graph_selfloop_edges = list(nx.selfloop_edges(conn_graph))
    
    # Number of core edges and periodic boundary edges
    core_m = np.shape(conn_core_edges)[0]
    pb_m = np.shape(conn_pb_edges)[0]

    # Plot formatting parameters
    plt_pad = plt_pad_prefactor * L
    core_tick_inc = core_tick_inc_prefactor * L
    min_core = -plt_pad
    max_core = L + plt_pad
    core_tick_steps = int(np.around((max_core-min_core)/core_tick_inc)) + 1

    xlim = np.asarray([min_core, max_core])
    ylim = np.asarray([min_core, max_core])

    xticks = np.linspace(min_core, max_core, core_tick_steps)
    yticks = np.linspace(min_core, max_core, core_tick_steps)

    xlabel = "x"
    ylabel = "y"

    grid_alpha = 0.25
    grid_zorder = 0

    # Core square box coordinates and preformating
    core_square = np.asarray(
        [
            [0, 0], [L, 0], [L, L], [0, L], [0, 0]
        ]
    )
    core_square_color = "red"
    core_square_linewidth = 0.5

    # Plot formatting parameters and preformatting
    zlim = np.asarray([min_core, max_core])
    zticks = np.linspace(min_core, max_core, core_tick_steps)
    zlabel = "z"

    # Core cube box coordinates and preformating
    core_cube = np.asarray(
        [
            [[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0], [0, 0, 0]],
            [[0, 0, L], [L, 0, L], [L, L, L], [0, L, L], [0, 0, L]],
            [[0, 0, 0], [L, 0, 0], [L, 0, L], [0, 0, L], [0, 0, 0]],
            [[L, 0, 0], [L, L, 0], [L, L, L], [L, 0, L], [L, 0, 0]],
            [[L, L, 0], [0, L, 0], [0, L, L], [L, L, L], [L, L, 0]],
            [[0, L, 0], [0, 0, 0], [0, 0, L], [0, L, L], [0, L, 0]]

        ]
    )
    core_cube_color = "red"
    core_cube_linewidth = 0.5

    if dim == 2:
        # Import two-dimension specific function
        from heterogeneous_spatial_networks_funcs import (
            dim_2_core_pb_edge_identification
        )

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots()
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            pb_node_0_x, pb_node_0_y, l_pb_edge = (
                dim_2_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y,
                    core_node_0_x, core_node_0_y, L)
            )
            pb_node_1_x, pb_node_1_y, l_pb_edge = (
                dim_2_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y,
                    core_node_1_x, core_node_1_y, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_0_x, pb_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_1_x, pb_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            pb_node_0_x, pb_node_0_y, l_pb_edge = (
                dim_2_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y,
                    core_node_0_x, core_node_0_y, L)
            )
            pb_node_1_x, pb_node_1_y, l_pb_edge = (
                dim_2_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y,
                    core_node_1_x, core_node_1_y, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_0_x, pb_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_1_x, pb_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots()
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots()
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:purple", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            ax.plot(
                edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, marker=marker, markersize=1.5,
                markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor)
        ax = dim_2_network_topology_axes_formatter(
            ax, core_square, core_square_color, core_square_linewidth,
            xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_filename)
        plt.close()
    elif dim == 3:
        # Import three-dimension specific function
        from heterogeneous_spatial_networks_funcs import (
            dim_3_core_pb_edge_identification
        )
        
        # Load fundamental z-dimensional graph constituents
        core_z = np.loadtxt(core_z_filename)

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5, alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                alpha=alpha)
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the spatial topology of the core and
        # periodic boundary nodes and edges. Here, core edges are
        # distinguished by purple lines, and periodic boundary edges are
        # distinguished by olive lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_1_x, core_node_1_y, core_node_1_z,
                    core_node_0_x, core_node_0_y, core_node_0_z, L)
            )
            pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
                dim_3_core_pb_edge_identification(
                    core_node_0_x, core_node_0_y, core_node_0_z,
                    core_node_1_x, core_node_1_y, core_node_1_z, L)
            )
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    pb_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    pb_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    pb_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                alpha=alpha)
            edge_x = np.asarray(
                [
                    pb_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    pb_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    pb_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            ax.plot(
                pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(core_pb_graph_colored_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_topology_filename)
        plt.close()

        # Plot of the core and periodic boundary cross-linkers and edges
        # for the graph capturing the periodic connections between the
        # core nodes. Here, core edges are distinguished by purple
        # lines, and periodic boundary edges are distinguished by olive
        # lines.
        fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
        for edge in range(core_m):
            core_node_0 = conn_core_edges[edge, 0]
            core_node_1 = conn_core_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        for edge in range(pb_m):
            core_node_0 = conn_pb_edges[edge, 0]
            core_node_1 = conn_pb_edges[edge, 1]
            alpha = aelp_network_edge_alpha(core_node_0, core_node_1, conn_graph)
            core_node_0_type = core_node_type[core_node_0]
            core_node_1_type = core_node_type[core_node_1]
            core_node_0_x = core_x[core_node_0]
            core_node_0_y = core_y[core_node_0]
            core_node_0_z = core_z[core_node_0]
            core_node_1_x = core_x[core_node_1]
            core_node_1_y = core_y[core_node_1]
            core_node_1_z = core_z[core_node_1]
            edge_x = np.asarray(
                [
                    core_node_0_x,
                    core_node_1_x
                ]
            )
            edge_y = np.asarray(
                [
                    core_node_0_y,
                    core_node_1_y
                ]
            )
            edge_z = np.asarray(
                [
                    core_node_0_z,
                    core_node_1_z
                ]
            )
            ax.plot(
                edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
                alpha=alpha)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_0, core_node_0_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_0_x, core_node_0_y, core_node_0_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
            marker, markerfacecolor, markeredgecolor = (
                aelp_network_core_node_marker_style(
                    core_node_1, core_node_1_type, conn_graph_selfloop_edges)
            )
            ax.plot(
                core_node_1_x, core_node_1_y, core_node_1_z, marker=marker,
                markersize=1.5, markerfacecolor=markerfacecolor,
                markeredgecolor=markeredgecolor)
        ax = dim_3_network_topology_axes_formatter(
            ax, core_cube, core_cube_color, core_cube_linewidth,
            xlim, ylim, zlim, xticks, yticks, zticks,
            xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
        fig.tight_layout()
        fig.savefig(conn_graph_colored_topology_filename)
        plt.close()
    
    # # Modified elastically-effective pruning procedure
    # from networkx.algorithms.connectivity.edge_kcomponents import bridge_components

    # # Create nx.MultiGraph and add nodes before edges
    # conn_core_graph = nx.MultiGraph()
    # conn_core_graph = add_nodes_from_numpy_array(conn_core_graph, core_nodes)
    # conn_core_graph = add_edges_from_numpy_array(conn_core_graph, conn_core_edges)

    # conn_pb_graph = nx.MultiGraph()
    # conn_pb_graph = add_nodes_from_numpy_array(conn_pb_graph, core_nodes)
    # conn_pb_graph = add_edges_from_numpy_array(conn_pb_graph, conn_pb_edges)

    # # Self-loop pruning procedure
    # if nx.number_of_selfloops(conn_graph) > 0:
    #     conn_graph.remove_edges_from(list(nx.selfloop_edges(conn_graph)))
    # if nx.number_of_selfloops(conn_core_graph) > 0:
    #     conn_core_graph.remove_edges_from(list(nx.selfloop_edges(conn_core_graph)))
    # if nx.number_of_selfloops(conn_pb_graph) > 0:
    #     conn_pb_graph.remove_edges_from(list(nx.selfloop_edges(conn_pb_graph)))
    
    # # Isolate largest/maximum bridge component in a nodewise fashion
    # ee_conn_graph_nodes = max(bridge_components(nx.Graph(conn_graph)), key=len)
    # # Extract largest/maximum bridge component subgraphs
    # ee_conn_core_graph = (
    #     conn_core_graph.subgraph(ee_conn_graph_nodes).copy()
    # )
    # ee_conn_pb_graph = (
    #     conn_pb_graph.subgraph(ee_conn_graph_nodes).copy()
    # )
    # # Extract edges
    # ee_conn_core_edges = np.asarray(list(ee_conn_core_graph.edges()), dtype=int)
    # ee_conn_pb_edges = np.asarray(list(ee_conn_pb_graph.edges()), dtype=int)
    
    # del ee_conn_graph_nodes
    
    # # Number of core edges and periodic boundary edges
    # ee_core_m = np.shape(ee_conn_core_edges)[0]
    # ee_pb_m = np.shape(ee_conn_pb_edges)[0]

    # if dim == 2:
    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the spatial topology of the core and
    #     # periodic boundary nodes and edges
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = (
    #             dim_2_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y,
    #                 core_node_0_x, core_node_0_y, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = (
    #             dim_2_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y,
    #                 core_node_1_x, core_node_1_y, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_0_x, pb_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_1_x, pb_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_core_pb_graph_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the spatial topology of the core and
    #     # periodic boundary nodes and edges. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         pb_node_0_x, pb_node_0_y, l_pb_edge = (
    #             dim_2_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y,
    #                 core_node_0_x, core_node_0_y, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, l_pb_edge = (
    #             dim_2_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y,
    #                 core_node_1_x, core_node_1_y, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_0_x, pb_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_1_x, pb_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_core_pb_graph_colored_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the periodic connections between the
    #     # core nodes
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_conn_graph_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the periodic connections between the
    #     # core nodes. Here, core edges are distinguished by purple
    #     # lines, and periodic boundary edges are distinguished by olive
    #     # lines.
    #     fig, ax = plt.subplots()
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:purple", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, color="tab:olive", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, marker=".", markersize=1.5,
    #             markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_2_network_topology_axes_formatter(
    #         ax, core_square, core_square_color, core_square_linewidth,
    #         xlim, ylim, xticks, yticks, xlabel, ylabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_conn_graph_colored_topology_filename)
    #     plt.close()
    # elif dim == 3:
    #     # Load fundamental z-dimensional graph constituents
    #     core_z = np.loadtxt(core_z_filename)

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the spatial topology of the core and
    #     # periodic boundary nodes and edges
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5, alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             alpha=alpha)
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks,
    #         xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_core_pb_graph_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the spatial topology of the core and
    #     # periodic boundary nodes and edges. Here, core edges are
    #     # distinguished by purple lines, and periodic boundary edges are
    #     # distinguished by olive lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black",
    #             markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         pb_node_0_x, pb_node_0_y, pb_node_0_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_1_x, core_node_1_y, core_node_1_z,
    #                 core_node_0_x, core_node_0_y, core_node_0_z, L)
    #         )
    #         pb_node_1_x, pb_node_1_y, pb_node_1_z, l_pb_edge = (
    #             dim_3_core_pb_edge_identification(
    #                 core_node_0_x, core_node_0_y, core_node_0_z,
    #                 core_node_1_x, core_node_1_y, core_node_1_z, L)
    #         )
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 pb_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 pb_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 pb_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             alpha=alpha)
    #         edge_x = np.asarray(
    #             [
    #                 pb_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 pb_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 pb_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_0_x, pb_node_0_y, pb_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             pb_node_1_x, pb_node_1_y, pb_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks,
    #         xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_core_pb_graph_colored_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the periodic connections between the
    #     # core nodes
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:blue", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks,
    #         xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_conn_graph_topology_filename)
    #     plt.close()

    #     # Plot of the core and periodic boundary cross-linkers and edges
    #     # for the graph capturing the periodic connections between the
    #     # core nodes. Here, core edges are distinguished by purple
    #     # lines, and periodic boundary edges are distinguished by olive
    #     # lines.
    #     fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    #     for edge in range(ee_core_m):
    #         core_node_0 = ee_conn_core_edges[edge, 0]
    #         core_node_1 = ee_conn_core_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_core_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:purple", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     for edge in range(ee_pb_m):
    #         core_node_0 = ee_conn_pb_edges[edge, 0]
    #         core_node_1 = ee_conn_pb_edges[edge, 1]
    #         alpha = aelp_network_edge_alpha(
    #             core_node_0, core_node_1, ee_conn_pb_graph)
    #         core_node_0_type = core_node_type[core_node_0]
    #         core_node_1_type = core_node_type[core_node_1]
    #         core_node_0_x = core_x[core_node_0]
    #         core_node_0_y = core_y[core_node_0]
    #         core_node_0_z = core_z[core_node_0]
    #         core_node_1_x = core_x[core_node_1]
    #         core_node_1_y = core_y[core_node_1]
    #         core_node_1_z = core_z[core_node_1]
    #         edge_x = np.asarray(
    #             [
    #                 core_node_0_x,
    #                 core_node_1_x
    #             ]
    #         )
    #         edge_y = np.asarray(
    #             [
    #                 core_node_0_y,
    #                 core_node_1_y
    #             ]
    #         )
    #         edge_z = np.asarray(
    #             [
    #                 core_node_0_z,
    #                 core_node_1_z
    #             ]
    #         )
    #         ax.plot(
    #             edge_x, edge_y, edge_z, color="tab:olive", linewidth=1.5,
    #             alpha=alpha)
    #         ax.plot(
    #             core_node_0_x, core_node_0_y, core_node_0_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #         ax.plot(
    #             core_node_1_x, core_node_1_y, core_node_1_z, marker=".",
    #             markersize=1.5, markerfacecolor="black", markeredgecolor="black")
    #     ax = dim_3_network_topology_axes_formatter(
    #         ax, core_cube, core_cube_color, core_cube_linewidth,
    #         xlim, ylim, zlim, xticks, yticks, zticks,
    #         xlabel, ylabel, zlabel, grid_alpha, grid_zorder)
    #     fig.tight_layout()
    #     fig.savefig(ee_conn_graph_colored_topology_filename)
    #     plt.close()

def run_aelp_network_topology_plotter(args):
    aelp_network_topology_plotter(*args)

def swidt_graph_k_density_histogram_plotter(
        k_dnstyhist_min: float,
        k_dnstyhist_max: float,
        k_dnstyhist_inc: float,
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        n: int,
        k: int,
        eta_n: float,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool) -> None:
    # k identifiers
    k_idntfr = "k"
    k_counts_idntfr = "k_counts"
    if elastically_effective:
        k_idntfr = "ee_" + k_idntfr
        k_counts_idntfr = "ee_" + k_counts_idntfr
    # Density histogram preformatting
    k_step = 1.0
    k_first_bin = 1.0
    k_last_bin = 8.0
    k_tick_inc = 1.0
    k_left_first_bin = k_first_bin - k_step/2
    k_right_last_bin = k_last_bin + k_step/2
    k_bins = np.arange(k_left_first_bin, k_right_last_bin+k_step, k_step)
    k_tick_steps = int(np.around((k_last_bin-k_first_bin)/k_tick_inc)) + 1
    xticks = np.linspace(k_first_bin, k_last_bin, k_tick_steps)
    k_dnstyhist_steps = (
        int(np.around((k_dnstyhist_max-k_dnstyhist_min)/k_dnstyhist_inc)) + 1
    )
    yticks = np.linspace(k_dnstyhist_min, k_dnstyhist_max, k_dnstyhist_steps)

    title = f"{dim:d}D, n = {n:d}, k_max = {k:d}, eta_n = {eta_n:0.3f}"
    filename_prefix = filename_str(network, date, batch, sample)
    k_dnstyhist_filename = (
        filename_prefix + "-" + k_idntfr + "-dnstyhist" + ".png"
    )

    graph_k_counts = np.zeros(8, dtype=int)
    for config in np.nditer(config_arr):
        for pruning in np.nditer(pruning_arr):
            graph_k_counts_filename_prefix = (
                filename_prefix + f"C{config:d}" + f"P{pruning:d}"
            )
            graph_k_counts_filename =  (
                graph_k_counts_filename_prefix + "-" + k_counts_idntfr + ".dat"
            )
            graph_k_counts += np.loadtxt(graph_k_counts_filename, dtype=int)

    plt.hist(
        k_bins[:-1], bins=k_bins, weights=graph_k_counts, density=True,
        color="tab:blue", edgecolor="black", zorder=3)
    plt.xticks(xticks)
    plt.xlabel("k", fontsize=16)
    plt.ylim([k_dnstyhist_min, k_dnstyhist_max])
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(k_dnstyhist_filename)
    plt.close()

def run_swidt_graph_k_density_histogram_plotter(args):
    swidt_graph_k_density_histogram_plotter(*args)

def swidt_graph_h_density_histogram_plotter(
        h_dnstyhist_min: float,
        h_dnstyhist_max: float,
        h_dnstyhist_inc: float,
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        n: int,
        k: int,
        eta_n: float,
        l_bound: int,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray) -> None:
    # Density histogram preformatting
    h_step = 1.0
    h_first_bin = 3.0
    h_last_bin = 1.0*l_bound
    h_tick_inc = 1.0
    h_left_first_bin = h_first_bin - h_step/2
    h_right_last_bin = h_last_bin + h_step/2
    h_bins = np.arange(h_left_first_bin, h_right_last_bin+h_step, h_step)
    h_tick_steps = int(np.around((h_last_bin-h_first_bin)/h_tick_inc)) + 1
    xticks = np.linspace(h_first_bin, h_last_bin, h_tick_steps)
    h_dnstyhist_steps = (
        int(np.around((h_dnstyhist_max-h_dnstyhist_min)/h_dnstyhist_inc)) + 1
    )
    yticks = np.linspace(h_dnstyhist_min, h_dnstyhist_max, h_dnstyhist_steps)

    title = f"{dim:d}D, n = {n:d}, k_max = {k:d}, eta_n = {eta_n:0.3f}"
    filename_prefix = filename_str(network, date, batch, sample)
    h_dnstyhist_filename = filename_prefix + "-h-dnstyhist" + ".png"

    graph_h_counts = np.zeros(l_bound, dtype=int)
    for config in np.nditer(config_arr):
        for pruning in np.nditer(pruning_arr):
            graph_h_counts_filename_prefix = (
                filename_prefix + f"C{config:d}" + f"P{pruning:d}"
            )
            graph_h_counts_filename =  (
                graph_h_counts_filename_prefix + "-h_counts" + ".dat"
            )
            graph_h_counts += np.loadtxt(graph_h_counts_filename, dtype=int)
    graph_h_counts = graph_h_counts[2:]
    
    plt.hist(
        h_bins[:-1], bins=h_bins, weights=graph_h_counts, density=True,
        color="tab:blue", edgecolor="black", zorder=3)
    plt.xticks(xticks)
    plt.xlabel("h", fontsize=16)
    plt.ylim([h_dnstyhist_min, h_dnstyhist_max])
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(h_dnstyhist_filename)
    plt.close()

def run_swidt_graph_h_density_histogram_plotter(args):
    swidt_graph_h_density_histogram_plotter(*args)

def swidt_graph_l_edges_variant_density_histogram_plotter(
        l_edges_vrnt_first_bin: float,
        l_edges_vrnt_last_bin: float,
        l_edges_vrnt_bin_inc: float,
        l_edges_vrnt_dnstyhist_min: float,
        l_edges_vrnt_dnstyhist_max: float,
        l_edges_vrnt_dnstyhist_inc: float,
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        n: int,
        k: int,
        eta_n: float,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool,
        l_edges_vrnt_idntfr: str) -> None:
    if dim == 2 and ((l_edges_vrnt_idntfr == "l_z_cmpnt_edges") or (l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges")):
        pass
    else:
        # Density histogram preformatting
        l_edges_vrnt_steps = (
            int(np.around((l_edges_vrnt_last_bin-l_edges_vrnt_first_bin)/l_edges_vrnt_bin_inc))
            + 1
        )
        l_edges_vrnt_bins = (
            np.linspace(
                l_edges_vrnt_first_bin,
                l_edges_vrnt_last_bin,
                l_edges_vrnt_steps)
        )
        xticks = (
            np.linspace(
                l_edges_vrnt_first_bin,
                l_edges_vrnt_last_bin,
                l_edges_vrnt_steps)
        )
        l_edges_vrnt_dnstyhist_steps = (
            int(np.around((l_edges_vrnt_dnstyhist_max-l_edges_vrnt_dnstyhist_min)/l_edges_vrnt_dnstyhist_inc))
            + 1
        )
        yticks = (
            np.linspace(
                l_edges_vrnt_dnstyhist_min,
                l_edges_vrnt_dnstyhist_max,
                l_edges_vrnt_dnstyhist_steps)
        )
        xlabel = ""
        if l_edges_vrnt_idntfr == "l_edges":
            xlabel = "l"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_edges":
            xlabel = "l_x"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_edges":
            xlabel = "l_y"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_edges":
            xlabel = "l_z"
        elif l_edges_vrnt_idntfr == "l_nrmlzd_edges":
            xlabel = "l/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_nrmlzd_edges":
            xlabel = "l_x/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_nrmlzd_edges":
            xlabel = "l_y/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges":
            xlabel = "l_z/(L*sqrt(dim))"

        if elastically_effective:
            l_edges_vrnt_idntfr = "ee_" + l_edges_vrnt_idntfr
            xlabel = "ee_" + xlabel
        
        title = f"{dim:d}D, n = {n:d}, k_max = {k:d}, eta_n = {eta_n:0.3f}"
        filename_prefix = filename_str(network, date, batch, sample)
        
        l_edges_vrnt_dnstyhist_filename = (
            filename_prefix + "-" + l_edges_vrnt_idntfr + "-dnstyhist" + ".png"
        )

        l_edges_vrnt = np.asarray([])
        for config in np.nditer(config_arr):
            for pruning in np.nditer(pruning_arr):
                l_edges_vrnt_filename_prefix = (
                    filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                )
                l_edges_vrnt_filename = (
                    l_edges_vrnt_filename_prefix + "-"
                    + l_edges_vrnt_idntfr + ".dat"
                )
                l_edges_vrnt = np.concatenate(
                    (l_edges_vrnt, np.loadtxt(l_edges_vrnt_filename)))

        plt.hist(
            l_edges_vrnt, bins=l_edges_vrnt_bins, density=True,
            color="tab:blue", edgecolor="black", zorder=3)
        plt.xticks(xticks)
        plt.xlabel(xlabel, fontsize=16)
        plt.ylim([l_edges_vrnt_dnstyhist_min, l_edges_vrnt_dnstyhist_max])
        plt.yticks(yticks)
        plt.title(title, fontsize=20)
        plt.grid(True, alpha=0.25, zorder=0)
        plt.tight_layout()
        plt.savefig(l_edges_vrnt_dnstyhist_filename)
        plt.close()

def run_swidt_graph_l_edges_variant_density_histogram_plotter(args):
    swidt_graph_l_edges_variant_density_histogram_plotter(*args)

def swidt_graph_k_christmas_tree_plot_plotter(
        k_xmastreeplt_min: float,
        k_xmastreeplt_max: float,
        k_xmastreeplt_inc: float,
        network: str,
        date: str,
        batch: str,
        unique_sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool) -> None:
    # k identifiers
    k_idntfr = "k"
    k_counts_idntfr = "k_counts"
    if elastically_effective:
        k_idntfr = "ee_" + k_idntfr
        k_counts_idntfr = "ee_" + k_counts_idntfr
    # Christmas tree plot preformatting
    k_num = np.shape(k_arr)[0]
    k_pad = 1.25
    k_ticks = k_pad * np.arange(k_num)
    ylim = np.asarray([k_xmastreeplt_min, k_xmastreeplt_max])
    k_xmastreeplt_steps = (
        int(np.around((k_xmastreeplt_max-k_xmastreeplt_min)/k_xmastreeplt_inc))
        + 1
    )
    yticks = (
        np.linspace(k_xmastreeplt_min, k_xmastreeplt_max, k_xmastreeplt_steps)
    )
    xlabel = "k_max"
    ylabel = "k"
    
    title = f"{dim:d}D, n = {n:d}, eta_n = {eta_n:0.3f}"
    filename_prefix = filename_str(network, date, batch, unique_sample)

    k_xmastreeplt_filename = (
        filename_prefix + "-" + k_idntfr + "-xmastreeplt" + ".png"
    )

    k_indx = 0
    for k in np.nditer(k_arr):
        sample = (
            int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
        )
        filename_prefix = filename_str(network, date, batch, sample)

        graph_k_counts = np.zeros(8, dtype=int)
        for config in np.nditer(config_arr):
            for pruning in np.nditer(pruning_arr):
                graph_k_counts_filename_prefix = (
                    filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                )
                graph_k_counts_filename =  (
                    graph_k_counts_filename_prefix + "-"
                    + k_counts_idntfr + ".dat"
                )
                graph_k_counts += np.loadtxt(graph_k_counts_filename, dtype=int)
        graph_k = np.asarray([], dtype=int)
        for graph_k_indx in range(8):
            graph_k = np.concatenate((graph_k, np.repeat(graph_k_indx+1, graph_k_counts[graph_k_indx])), dtype=int)
        graph_k_vals, graph_k_counts = np.unique(graph_k, return_counts=True)
        graph_k_dnsty = graph_k_counts / np.sum(graph_k_counts)

        plt.barh(
            graph_k_vals, graph_k_dnsty, height=1, left=(k_pad*k_indx),
            color="tab:blue", zorder=3)
        plt.barh(
            graph_k_vals, -graph_k_dnsty, height=1, left=(k_pad*k_indx),
            color="tab:blue", zorder=3)
        
        k_indx += 1
    
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_ticks, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(k_xmastreeplt_filename)
    plt.close()

def run_swidt_graph_k_christmas_tree_plot_plotter(args):
    swidt_graph_k_christmas_tree_plot_plotter(*args)

def swidt_graph_k_dim_christmas_tree_plot_plotter(
        k_xmastreeplt_min: float,
        k_xmastreeplt_max: float,
        k_xmastreeplt_inc: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool) -> None:
    # k identifiers
    k_idntfr = "k"
    k_counts_idntfr = "k_counts"
    if elastically_effective:
        k_idntfr = "ee_" + k_idntfr
        k_counts_idntfr = "ee_" + k_counts_idntfr
    # Christmas tree plot preformatting
    k_num = np.shape(k_arr)[0]
    k_pad = 1.25
    k_ticks = k_pad * np.arange(k_num)
    ylim = np.asarray([k_xmastreeplt_min, k_xmastreeplt_max])
    k_xmastreeplt_steps = (
        int(np.around((k_xmastreeplt_max-k_xmastreeplt_min)/k_xmastreeplt_inc))
        + 1
    )
    yticks = (
        np.linspace(k_xmastreeplt_min, k_xmastreeplt_max, k_xmastreeplt_steps)
    )
    xlabel = "k_max"
    ylabel = "k"

    title = f"{dim:d}D"
    filename_prefix = filepath_str(network) + f"{date}{batch}-dim_{dim:d}"

    k_xmastreeplt_filename = (
        filename_prefix + "-" + k_idntfr + "-xmastreeplt" + ".png"
    )

    for n in np.nditer(n_arr):
        for eta_n in np.nditer(eta_n_arr):
            k_indx = 0
            for k in np.nditer(k_arr):
                sample = (
                    int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                )
                filename_prefix = filename_str(network, date, batch, sample)

                graph_k_counts = np.zeros(8, dtype=int)
                for config in np.nditer(config_arr):
                    for pruning in np.nditer(pruning_arr):
                        graph_k_counts_filename_prefix = (
                            filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                        )
                        graph_k_counts_filename = (
                            graph_k_counts_filename_prefix + "-"
                            + k_counts_idntfr + ".dat"
                        )
                        graph_k_counts += np.loadtxt(graph_k_counts_filename, dtype=int)
                graph_k = np.asarray([], dtype=int)
                for graph_k_indx in range(8):
                    graph_k = np.concatenate((graph_k, np.repeat(graph_k_indx+1, graph_k_counts[graph_k_indx])), dtype=int)
                graph_k_vals, graph_k_counts = np.unique(graph_k, return_counts=True)
                graph_k_dnsty = graph_k_counts / np.sum(graph_k_counts)

                plt.barh(
                    graph_k_vals, graph_k_dnsty, height=1, left=(k_pad*k_indx),
                    color="tab:blue", alpha=0.25, zorder=3)
                plt.barh(
                    graph_k_vals, -graph_k_dnsty, height=1, left=(k_pad*k_indx),
                    color="tab:blue", alpha=0.25, zorder=3)
                
                k_indx += 1
    
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_ticks, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(k_xmastreeplt_filename)
    plt.close()

def run_swidt_graph_k_dim_christmas_tree_plot_plotter(args):
    swidt_graph_k_dim_christmas_tree_plot_plotter(*args)

def swidt_graph_k_dim_dist_stats_plotter(
        k_dist_stats_plt_min: float,
        k_dist_stats_plt_max: float,
        k_dist_stats_plt_inc: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool) -> None:
    # k identifiers
    k_idntfr = "k"
    k_counts_idntfr = "k_counts"
    if elastically_effective:
        k_idntfr = "ee_" + k_idntfr
        k_counts_idntfr = "ee_" + k_counts_idntfr
    # Statistical distributions plot preformatting
    q1 = 0.25
    q3 = 0.75
    q1_q3_arr = np.asarray([q1, q3])
    k_num = np.shape(k_arr)[0]
    ylim = np.asarray([k_dist_stats_plt_min, k_dist_stats_plt_max])
    k_dist_stats_plt_steps = (
        int(np.around((k_dist_stats_plt_max-k_dist_stats_plt_min)/k_dist_stats_plt_inc))
        + 1
    )
    yticks = (
        np.linspace(
            k_dist_stats_plt_min,
            k_dist_stats_plt_max,
            k_dist_stats_plt_steps)
    )
    xlabel = "k_max"
    ylabel = "k"

    title = f"{dim:d}D"
    filename_prefix = filepath_str(network) + f"{date}{batch}-dim_{dim:d}"

    k_dist_stats_plt_filename = (
        filename_prefix + "-" + k_idntfr + "-dist_stats_plt" + ".png"
    )

    graph_k_min_arr = np.empty(k_num)
    graph_k_q1_arr = np.empty(k_num)
    graph_k_mean_arr = np.empty(k_num)
    graph_k_q3_arr = np.empty(k_num)
    graph_k_max_arr = np.empty(k_num)
    k_indx = 0
    for k in np.nditer(k_arr):
        graph_k_counts = np.zeros(8, dtype=int)
        for n in np.nditer(n_arr):
            for eta_n in np.nditer(eta_n_arr):
                sample = (
                    int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                )
                filename_prefix = filename_str(network, date, batch, sample)

                for config in np.nditer(config_arr):
                    for pruning in np.nditer(pruning_arr):
                        graph_k_counts_filename_prefix = (
                            filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                        )
                        graph_k_counts_filename =  (
                            graph_k_counts_filename_prefix + "-"
                            + k_counts_idntfr + ".dat"
                        )
                        graph_k_counts += np.loadtxt(graph_k_counts_filename, dtype=int)
        graph_k = np.asarray([], dtype=int)
        for graph_k_indx in range(8):
            graph_k = np.concatenate((graph_k, np.repeat(graph_k_indx+1, graph_k_counts[graph_k_indx])), dtype=int)
        graph_k_q1_val, graph_k_q3_val = np.quantile(graph_k, q1_q3_arr)
        graph_k_min_arr[k_indx] = np.min(graph_k)
        graph_k_q1_arr[k_indx] = graph_k_q1_val
        graph_k_mean_arr[k_indx] = np.mean(graph_k)
        graph_k_q3_arr[k_indx] = graph_k_q3_val
        graph_k_max_arr[k_indx] = np.max(graph_k)
        k_indx += 1
    
    del graph_k

    plt.fill_between(
        k_arr, graph_k_min_arr, graph_k_max_arr, color="skyblue",
        alpha=0.25)
    plt.fill_between(
        k_arr, graph_k_q1_arr, graph_k_q3_arr, color="steelblue",
        alpha=0.25)
    plt.plot(k_arr, graph_k_mean_arr, linestyle="-", color="tab:blue")
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_arr, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(k_dist_stats_plt_filename)
    plt.close()

def run_swidt_graph_k_dim_dist_stats_plotter(args):
    swidt_graph_k_dim_dist_stats_plotter(*args)

def swidt_graph_h_christmas_tree_plot_plotter(
        network: str,
        date: str,
        batch: str,
        unique_sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray) -> None:
    # Christmas tree plot preformatting
    k_num = np.shape(k_arr)[0]
    k_pad = 1.25
    k_ticks = k_pad * np.arange(k_num)
    h_xmastreeplt_min = 2.0
    h_xmastreeplt_max = 26.0
    h_xmastreeplt_inc = 1.0
    ylim = np.asarray([h_xmastreeplt_min, h_xmastreeplt_max])
    h_xmastreeplt_steps = (
        int(np.around((h_xmastreeplt_max-h_xmastreeplt_min)/h_xmastreeplt_inc))
        + 1
    )
    yticks = (
        np.linspace(h_xmastreeplt_min, h_xmastreeplt_max, h_xmastreeplt_steps)
    )
    xlabel = "k_max"
    ylabel = "h"
    
    title = f"{dim:d}D, n = {n:d}, eta_n = {eta_n:0.3f}"
    filename_prefix = filename_str(network, date, batch, unique_sample)

    h_xmastreeplt_filename = filename_prefix + "-h-xmastreeplt" + ".png"

    l_bound = 0
    k_indx = 0
    for k in np.nditer(k_arr):
        sample = (
            int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
        )
        filename_prefix = filename_str(network, date, batch, sample)

        if k == 3: l_bound = 25
        elif k == 4: l_bound = 15
        elif k == 5: l_bound = 14
        elif k == 6: l_bound = 12
        elif k == 7: l_bound = 10
        elif k == 8: l_bound = 10

        graph_h_counts = np.zeros(l_bound, dtype=int)
        for config in np.nditer(config_arr):
            for pruning in np.nditer(pruning_arr):
                graph_h_counts_filename_prefix = (
                    filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                )
                graph_h_counts_filename =  (
                    graph_h_counts_filename_prefix + "-h_counts" + ".dat"
                )
                graph_h_counts += np.loadtxt(graph_h_counts_filename, dtype=int)
        graph_h_counts = graph_h_counts[2:]
        graph_h = np.asarray([], dtype=int)
        for graph_h_indx in range(l_bound-2):
            graph_h = np.concatenate((graph_h, np.repeat(graph_h_indx+3, graph_h_counts[graph_h_indx])), dtype=int)
        graph_h_vals, graph_h_counts = np.unique(graph_h, return_counts=True)
        graph_h_dnsty = graph_h_counts / np.sum(graph_h_counts)

        plt.barh(
            graph_h_vals, graph_h_dnsty, height=1, left=(k_pad*k_indx),
            color="tab:blue", zorder=3)
        plt.barh(
            graph_h_vals, -graph_h_dnsty, height=1, left=(k_pad*k_indx),
            color="tab:blue", zorder=3)
        
        k_indx += 1
    
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_ticks, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(h_xmastreeplt_filename)
    plt.close()

def run_swidt_graph_h_christmas_tree_plot_plotter(args):
    swidt_graph_h_christmas_tree_plot_plotter(*args)

def swidt_graph_h_dim_christmas_tree_plot_plotter(
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray) -> None:
    # Christmas tree plot preformatting
    k_num = np.shape(k_arr)[0]
    k_pad = 1.25
    k_ticks = k_pad * np.arange(k_num)
    h_xmastreeplt_min = 2.0
    h_xmastreeplt_max = 26.0
    h_xmastreeplt_inc = 1.0
    ylim = np.asarray([h_xmastreeplt_min, h_xmastreeplt_max])
    h_xmastreeplt_steps = (
        int(np.around((h_xmastreeplt_max-h_xmastreeplt_min)/h_xmastreeplt_inc))
        + 1
    )
    yticks = (
        np.linspace(h_xmastreeplt_min, h_xmastreeplt_max, h_xmastreeplt_steps)
    )
    xlabel = "k_max"
    ylabel = "h"

    title = f"{dim:d}D"
    filename_prefix = filepath_str(network) + f"{date}{batch}-dim_{dim:d}"

    h_xmastreeplt_filename = filename_prefix + "-h-xmastreeplt" + ".png"

    for n in np.nditer(n_arr):
        for eta_n in np.nditer(eta_n_arr):
            l_bound = 0
            k_indx = 0
            for k in np.nditer(k_arr):
                sample = (
                    int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                )
                filename_prefix = filename_str(network, date, batch, sample)

                if k == 3: l_bound = 25
                elif k == 4: l_bound = 15
                elif k == 5: l_bound = 14
                elif k == 6: l_bound = 12
                elif k == 7: l_bound = 10
                elif k == 8: l_bound = 10

                graph_h_counts = np.zeros(l_bound, dtype=int)
                for config in np.nditer(config_arr):
                    for pruning in np.nditer(pruning_arr):
                        graph_h_counts_filename_prefix = (
                            filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                        )
                        graph_h_counts_filename = (
                            graph_h_counts_filename_prefix + "-h_counts" + ".dat"
                        )
                        graph_h_counts += np.loadtxt(graph_h_counts_filename, dtype=int)
                graph_h_counts = graph_h_counts[2:]
                graph_h = np.asarray([], dtype=int)
                for graph_h_indx in range(l_bound-2):
                    graph_h = np.concatenate((graph_h, np.repeat(graph_h_indx+3, graph_h_counts[graph_h_indx])), dtype=int)
                graph_h_vals, graph_h_counts = np.unique(graph_h, return_counts=True)
                graph_h_dnsty = graph_h_counts / np.sum(graph_h_counts)

                plt.barh(
                    graph_h_vals, graph_h_dnsty, height=1, left=(k_pad*k_indx),
                    color="tab:blue", alpha=0.25, zorder=3)
                plt.barh(
                    graph_h_vals, -graph_h_dnsty, height=1, left=(k_pad*k_indx),
                    color="tab:blue", alpha=0.25, zorder=3)
                
                k_indx += 1
    
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_ticks, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(h_xmastreeplt_filename)
    plt.close()

def run_swidt_graph_h_dim_christmas_tree_plot_plotter(args):
    swidt_graph_h_dim_christmas_tree_plot_plotter(*args)

def swidt_graph_h_dim_dist_stats_plotter(
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        n_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        k_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray) -> None:
    # Statistical distributions plot preformatting
    q1 = 0.25
    q3 = 0.75
    q1_q3_arr = np.asarray([q1, q3])
    k_num = np.shape(k_arr)[0]
    h_dist_stats_plt_min = 2.0
    h_dist_stats_plt_max = 26.0
    h_dist_stats_plt_inc = 1.0
    ylim = np.asarray([h_dist_stats_plt_min, h_dist_stats_plt_max])
    h_dist_stats_plt_steps = (
        int(np.around((h_dist_stats_plt_max-h_dist_stats_plt_min)/h_dist_stats_plt_inc))
        + 1
    )
    yticks = (
        np.linspace(
            h_dist_stats_plt_min,
            h_dist_stats_plt_max,
            h_dist_stats_plt_steps)
    )
    xlabel = "k_max"
    ylabel = "h"

    title = f"{dim:d}D"
    filename_prefix = (
        filepath_str(network) + f"{date}{batch}-dim_{dim:d}"
    )

    h_dist_stats_plt_filename = filename_prefix + "-h-dist_stats_plt" + ".png"

    graph_h_min_arr = np.empty(k_num)
    graph_h_q1_arr = np.empty(k_num)
    graph_h_mean_arr = np.empty(k_num)
    graph_h_q3_arr = np.empty(k_num)
    graph_h_max_arr = np.empty(k_num)
    
    l_bound = 0
    k_indx = 0
    for k in np.nditer(k_arr):
        if k == 3: l_bound = 25
        elif k == 4: l_bound = 15
        elif k == 5: l_bound = 14
        elif k == 6: l_bound = 12
        elif k == 7: l_bound = 10
        elif k == 8: l_bound = 10
        
        graph_h_counts = np.zeros(l_bound, dtype=int)
        for n in np.nditer(n_arr):
            for eta_n in np.nditer(eta_n_arr):
                sample = (
                    int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                )
                filename_prefix = filename_str(network, date, batch, sample)

                for config in np.nditer(config_arr):
                    for pruning in np.nditer(pruning_arr):
                        graph_h_counts_filename_prefix = (
                            filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                        )
                        graph_h_counts_filename =  (
                            graph_h_counts_filename_prefix + "-h_counts" + ".dat"
                        )
                        graph_h_counts += np.loadtxt(graph_h_counts_filename, dtype=int)
        graph_h_counts = graph_h_counts[2:]
        graph_h = np.asarray([], dtype=int)
        for graph_h_indx in range(l_bound-2):
            graph_h = np.concatenate((graph_h, np.repeat(graph_h_indx+3, graph_h_counts[graph_h_indx])), dtype=int)
        graph_h_q1_val, graph_h_q3_val = np.quantile(graph_h, q1_q3_arr)
        graph_h_min_arr[k_indx] = np.min(graph_h)
        graph_h_q1_arr[k_indx] = graph_h_q1_val
        graph_h_mean_arr[k_indx] = np.mean(graph_h)
        graph_h_q3_arr[k_indx] = graph_h_q3_val
        graph_h_max_arr[k_indx] = np.max(graph_h)
        k_indx += 1
    
    del graph_h

    plt.fill_between(
        k_arr, graph_h_min_arr, graph_h_max_arr, color="skyblue",
        alpha=0.25)
    plt.fill_between(
        k_arr, graph_h_q1_arr, graph_h_q3_arr, color="steelblue",
        alpha=0.25)
    plt.plot(k_arr, graph_h_mean_arr, linestyle="-", color="tab:blue")
    plt.xlabel(xlabel, fontsize=16)
    plt.xticks(ticks=k_arr, labels=k_arr)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.title(title, fontsize=20)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.tight_layout()
    plt.savefig(h_dist_stats_plt_filename)
    plt.close()

def run_swidt_graph_h_dim_dist_stats_plotter(args):
    swidt_graph_h_dim_dist_stats_plotter(*args)

def swidt_graph_l_edges_variant_violinplot_plotter(
        l_edges_vrnt_vlnplt_min: float,
        l_edges_vrnt_vlnplt_max: float,
        l_edges_vrnt_vlnplt_inc: float,
        network: str,
        date: str,
        batch: str,
        unique_sample: int,
        dim: int,
        b: float,
        k: int,
        eta_n: float,
        n_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool,
        l_edges_vrnt_idntfr: str) -> None:
    if dim == 2 and ((l_edges_vrnt_idntfr == "l_z_cmpnt_edges") or (l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges")):
        pass
    else:
        # Violinplot preformatting
        n_num = np.shape(n_arr)[0]
        n_ticks = 1 + np.arange(n_num)
        ylim = np.asarray([l_edges_vrnt_vlnplt_min, l_edges_vrnt_vlnplt_max])
        l_edges_vrnt_vlnplt_steps = (
            int(np.around((l_edges_vrnt_vlnplt_max-l_edges_vrnt_vlnplt_min)/l_edges_vrnt_vlnplt_inc))
            + 1
        )
        yticks = (
            np.linspace(
                l_edges_vrnt_vlnplt_min,
                l_edges_vrnt_vlnplt_max,
                l_edges_vrnt_vlnplt_steps)
        )
        xlabel = "n"
        ylabel = ""
        if l_edges_vrnt_idntfr == "l_edges":
            ylabel = "l"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_edges":
            ylabel = "l_x"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_edges":
            ylabel = "l_y"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_edges":
            ylabel = "l_z"
        elif l_edges_vrnt_idntfr == "l_nrmlzd_edges":
            ylabel = "l/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_nrmlzd_edges":
            ylabel = "l_x/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_nrmlzd_edges":
            ylabel = "l_y/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges":
            ylabel = "l_z/(L*sqrt(dim))"

        if elastically_effective:
            l_edges_vrnt_idntfr = "ee_" + l_edges_vrnt_idntfr
            ylabel = "ee_" + ylabel
        
        title = f"{dim:d}D, k_max = {k:d}, eta_n = {eta_n:0.3f}"
        filename_prefix = filename_str(network, date, batch, unique_sample)
        
        l_edges_vrnt_vlnplt_filename = (
            filename_prefix + "-" + l_edges_vrnt_idntfr + "-vlnplt" + ".png"
        )

        l_edges_vrnt_list = []
        n_indx = 0
        for n in np.nditer(n_arr):
            sample = (
                int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
            )
            filename_prefix = filename_str(network, date, batch, sample)

            l_edges_vrnt = np.asarray([])
            for config in np.nditer(config_arr):
                for pruning in np.nditer(pruning_arr):
                    l_edges_vrnt_filename_prefix = (
                        filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                    )
                    l_edges_vrnt_filename = (
                        l_edges_vrnt_filename_prefix + "-"
                        + l_edges_vrnt_idntfr + ".dat"
                    )
                    l_edges_vrnt = np.concatenate(
                        (l_edges_vrnt, np.loadtxt(l_edges_vrnt_filename)))
            l_edges_vrnt_list.append(l_edges_vrnt)
            n_indx += 1
        
        l_edges_vrnt_vp = plt.violinplot(
            l_edges_vrnt_list, showextrema=False)
        for vp in l_edges_vrnt_vp["bodies"]:
            vp.set_facecolor("tab:blue")
            vp.set_edgecolor("tab:blue")
        plt.xlabel(xlabel, fontsize=16)
        plt.xticks(ticks=n_ticks, labels=n_arr)
        plt.ylabel(ylabel, fontsize=16)
        plt.ylim(ylim)
        plt.yticks(yticks)
        plt.title(title, fontsize=20)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(l_edges_vrnt_vlnplt_filename)
        plt.close()

def run_swidt_graph_l_edges_variant_violinplot_plotter(args):
    swidt_graph_l_edges_variant_violinplot_plotter(*args)

def swidt_graph_l_edges_variant_dim_violinplot_plotter(
        l_edges_vrnt_vlnplt_min: float,
        l_edges_vrnt_vlnplt_max: float,
        l_edges_vrnt_vlnplt_inc: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        k_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        n_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool,
        l_edges_vrnt_idntfr: str) -> None:
    if dim == 2 and ((l_edges_vrnt_idntfr == "l_z_cmpnt_edges") or (l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges")):
        pass
    else:
        # Violinplot preformatting
        n_num = np.shape(n_arr)[0]
        n_ticks = 1 + np.arange(n_num)
        ylim = np.asarray([l_edges_vrnt_vlnplt_min, l_edges_vrnt_vlnplt_max])
        l_edges_vrnt_vlnplt_steps = (
            int(np.around((l_edges_vrnt_vlnplt_max-l_edges_vrnt_vlnplt_min)/l_edges_vrnt_vlnplt_inc))
            + 1
        )
        yticks = (
            np.linspace(
                l_edges_vrnt_vlnplt_min,
                l_edges_vrnt_vlnplt_max,
                l_edges_vrnt_vlnplt_steps)
        )
        xlabel = "n"
        ylabel = ""
        if l_edges_vrnt_idntfr == "l_edges":
            ylabel = "l"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_edges":
            ylabel = "l_x"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_edges":
            ylabel = "l_y"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_edges":
            ylabel = "l_z"
        elif l_edges_vrnt_idntfr == "l_nrmlzd_edges":
            ylabel = "l/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_nrmlzd_edges":
            ylabel = "l_x/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_nrmlzd_edges":
            ylabel = "l_y/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges":
            ylabel = "l_z/(L*sqrt(dim))"

        if elastically_effective:
            l_edges_vrnt_idntfr = "ee_" + l_edges_vrnt_idntfr
            ylabel = "ee_" + ylabel
        
        title = f"{dim:d}D"
        filename_prefix = filepath_str(network) + f"{date}{batch}-dim_{dim:d}"
        
        l_edges_vrnt_vlnplt_filename = (
            filename_prefix + "-" + l_edges_vrnt_idntfr + "-vlnplt" + ".png"
        )

        for k in np.nditer(k_arr):
            for eta_n in np.nditer(eta_n_arr):
                l_edges_vrnt_list = []
                n_indx = 0
                for n in np.nditer(n_arr):
                    sample = (
                        int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                    )
                    filename_prefix = filename_str(network, date, batch, sample)

                    l_edges_vrnt = np.asarray([])
                    for config in np.nditer(config_arr):
                        for pruning in np.nditer(pruning_arr):
                            l_edges_vrnt_filename_prefix = (
                                filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                            )
                            l_edges_vrnt_filename = (
                                l_edges_vrnt_filename_prefix + "-"
                                + l_edges_vrnt_idntfr + ".dat"
                            )
                            l_edges_vrnt = np.concatenate(
                                (l_edges_vrnt, np.loadtxt(l_edges_vrnt_filename)))
                    l_edges_vrnt_list.append(l_edges_vrnt)
                    n_indx += 1
                
                l_edges_vrnt_vp = plt.violinplot(
                    l_edges_vrnt_list, showextrema=False)
                for vp in l_edges_vrnt_vp["bodies"]:
                    vp.set_facecolor("tab:blue")
                    vp.set_edgecolor("tab:blue")
                    vp.set_alpha(0.25)
        plt.xlabel(xlabel, fontsize=16)
        plt.xticks(ticks=n_ticks, labels=n_arr)
        plt.ylabel(ylabel, fontsize=16)
        plt.ylim(ylim)
        plt.yticks(yticks)
        plt.title(title, fontsize=20)
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(l_edges_vrnt_vlnplt_filename)
        plt.close()

def run_swidt_graph_l_edges_variant_dim_violinplot_plotter(args):
    swidt_graph_l_edges_variant_dim_violinplot_plotter(*args)

def swidt_graph_l_edges_variant_dim_dist_stats_plotter(
        l_edges_vrnt_dist_stats_plt_min: float,
        l_edges_vrnt_dist_stats_plt_max: float,
        l_edges_vrnt_dist_stats_plt_inc: float,
        network: str,
        date: str,
        batch: str,
        dim: int,
        b: float,
        k_arr: np.ndarray,
        eta_n_arr: np.ndarray,
        n_arr: np.ndarray,
        params_arr: np.ndarray,
        config_arr: np.ndarray,
        pruning_arr: np.ndarray,
        elastically_effective: bool,
        l_edges_vrnt_idntfr: str) -> None:
    if dim == 2 and ((l_edges_vrnt_idntfr == "l_z_cmpnt_edges") or (l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges")):
        pass
    else:
        # Statistical distributions plot preformatting
        q1 = 0.25
        q3 = 0.75
        q1_q3_arr = np.asarray([q1, q3])
        n_num = np.shape(n_arr)[0]
        ylim = np.asarray(
            [l_edges_vrnt_dist_stats_plt_min, l_edges_vrnt_dist_stats_plt_max])
        l_edges_vrnt_dist_stats_plt_steps = (
            int(np.around((l_edges_vrnt_dist_stats_plt_max-l_edges_vrnt_dist_stats_plt_min)/l_edges_vrnt_dist_stats_plt_inc))
            + 1
        )
        n_ticks = np.arange(n_num)
        yticks = (
            np.linspace(
                l_edges_vrnt_dist_stats_plt_min,
                l_edges_vrnt_dist_stats_plt_max,
                l_edges_vrnt_dist_stats_plt_steps)
        )
        xlabel = "n"
        ylabel = ""
        if l_edges_vrnt_idntfr == "l_edges":
            ylabel = "l"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_edges":
            ylabel = "l_x"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_edges":
            ylabel = "l_y"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_edges":
            ylabel = "l_z"
        elif l_edges_vrnt_idntfr == "l_nrmlzd_edges":
            ylabel = "l/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_x_cmpnt_nrmlzd_edges":
            ylabel = "l_x/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_y_cmpnt_nrmlzd_edges":
            ylabel = "l_y/(L*sqrt(dim))"
        elif l_edges_vrnt_idntfr == "l_z_cmpnt_nrmlzd_edges":
            ylabel = "l_z/(L*sqrt(dim))"

        if elastically_effective:
            l_edges_vrnt_idntfr = "ee_" + l_edges_vrnt_idntfr
            ylabel = "ee_" + ylabel
        
        title = f"{dim:d}D"
        filename_prefix = filepath_str(network) + f"{date}{batch}-dim_{dim:d}"
        
        l_edges_vrnt_dist_stats_plt_filename = (
            filename_prefix + "-" + l_edges_vrnt_idntfr
            + "-dist_stats_plt" + ".png"
        )

        l_edges_vrnt_min_arr = np.empty(n_num)
        l_edges_vrnt_q1_arr = np.empty(n_num)
        l_edges_vrnt_mean_arr = np.empty(n_num)
        l_edges_vrnt_q3_arr = np.empty(n_num)
        l_edges_vrnt_max_arr = np.empty(n_num)
        n_indx = 0
        for n in np.nditer(n_arr):
            l_edges_vrnt = np.asarray([])
            for k in np.nditer(k_arr):
                for eta_n in np.nditer(eta_n_arr):
                    sample = (
                        int(np.where((params_arr == (dim, b, n, k, eta_n)).all(axis=1))[0][0])
                    )
                    filename_prefix = filename_str(network, date, batch, sample)

                    for config in np.nditer(config_arr):
                        for pruning in np.nditer(pruning_arr):
                            l_edges_vrnt_filename_prefix = (
                                filename_prefix + f"C{config:d}" + f"P{pruning:d}"
                            )
                            l_edges_vrnt_filename = (
                                l_edges_vrnt_filename_prefix + "-"
                                + l_edges_vrnt_idntfr + ".dat"
                            )
                            l_edges_vrnt = np.concatenate(
                                (l_edges_vrnt, np.loadtxt(l_edges_vrnt_filename)))
            l_edges_vrnt_q1_val, l_edges_vrnt_q3_val = (
                np.quantile(l_edges_vrnt, q1_q3_arr)
            )
            l_edges_vrnt_min_arr[n_indx] = (
                np.min(l_edges_vrnt)
            )
            l_edges_vrnt_q1_arr[n_indx] = (
                l_edges_vrnt_q1_val
            )
            l_edges_vrnt_mean_arr[n_indx] = (
                np.mean(l_edges_vrnt)
            )
            l_edges_vrnt_q3_arr[n_indx] = (
                l_edges_vrnt_q3_val
            )
            l_edges_vrnt_max_arr[n_indx] = (
                np.max(l_edges_vrnt)
            )
            n_indx += 1
        
        del l_edges_vrnt

        plt.fill_between(
            n_arr, l_edges_vrnt_min_arr,
            l_edges_vrnt_max_arr, color="skyblue", alpha=0.25)
        plt.fill_between(
            n_arr, l_edges_vrnt_q1_arr,
            l_edges_vrnt_q3_arr, color="steelblue", alpha=0.25)
        plt.plot(
            n_arr, l_edges_vrnt_mean_arr, linestyle="-",
            color="tab:blue")
        plt.xscale("log")
        plt.xlabel(xlabel, fontsize=16)
        plt.ylabel(ylabel, fontsize=16)
        plt.ylim(ylim)
        plt.yticks(yticks)
        plt.title(title, fontsize=20)
        plt.grid(True, alpha=0.25, zorder=0)
        plt.tight_layout()
        plt.savefig(l_edges_vrnt_dist_stats_plt_filename)
        plt.close()

def run_swidt_graph_l_edges_variant_dim_dist_stats_plotter(args):
    swidt_graph_l_edges_variant_dim_dist_stats_plotter(*args)