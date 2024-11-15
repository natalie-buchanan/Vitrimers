# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:20:35 2024

@author: Natalie.Buchanan
"""
# %%
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import file_functions
from beads_from_edge_3d import create_atom_list

# %%
def unwrap_nodes(start, end, box_size):
    """Wraps second point miminize distance to first point"""
    for i in range(len(start)):
        diff = np.abs(end[i] - start[i])
        if diff > (box_size-diff):
            end[i] -= box_size
    return end


def wrap_coords(first_point, box_size):
    """Wrap point into simulation box."""
    v1 = np.array(first_point, dtype='float64')
    z1 = v1.copy()
    while np.any(np.less(z1, 0)):
        z1[z1 < 0] += box_size
    while np.any(np.greater(z1, box_size)):
        z1[z1 > box_size] -= box_size
    return z1


def upside_down_trapezoid(v1, v4, number_of_beads, short_base):
    """Defines vertices and side lengths when node distance is shortest side"""
    possible_shapes = []
    for long_base in np.arange(1, (number_of_beads + short_base)/2, 2):
        side = (number_of_beads - long_base)/2
        if (long_base % 1 == 0) and (side % 1 == 0):
            possible_shapes.append([long_base, side])
    long_base, side = random.choice(possible_shapes)

    height = 1/2*np.sqrt((4*(side**2)) - ((long_base-short_base)**2))
    alpha = np.arcsin(height/side)

    base_vector = v4 - v1
    base_length = np.linalg.norm(base_vector)

    if abs(base_length - short_base) > 1e-6:
        print("Error: provided base length does not match side_lengths[0]")
        return None

    normal_vector = np.cross(base_vector, np.array([0, 0, 1]))
    if np.allclose(normal_vector, np.array([0, 0, 0])):
        normal_vector = np.cross(base_vector, np.array([0, 1, 0]))

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    delta = np.cos(alpha)
    v2 = v1 + delta*base_vector + height*normal_vector
    v3 = v4 + delta*base_vector + height*normal_vector
    points = [v1, v2, v3, v4]
    lengths = [short_base, long_base, side]
    return points, lengths


def trapezoid(v1, v4, number_of_beads, long_base):
    """Defines vertices and side lengths when node distance is shortest side"""
    possible_shapes = []
    if not number_of_beads % 2:
        within = np.arange(1, long_base, 2)
    else:
        within = np.arange(0, long_base, 2)
    if len(within) == 0:
        within = [0]
    for short_base in within:
        side = (number_of_beads - 1 - short_base)/2
        if (short_base % 1 == 0) and (side % 1 == 0):
            possible_shapes.append([short_base, side])
    if len(possible_shapes) == 0:
        print("HELP")
    else:
        short_base, side = random.choice(possible_shapes)

    height = 1/2*np.sqrt((4*(side**2)) - ((long_base-short_base)**2))
    alpha = np.arcsin(height/side)

    base_vector = v4 - v1
    base_length = np.linalg.norm(base_vector)

    if abs(base_length - long_base) > 1e-6:
        print("Error: provided base length does not match side_lengths[0]")
        return None

    normal_vector = np.cross(base_vector, np.array([0, 0, 1]))
    if np.allclose(normal_vector, np.array([0, 0, 0])):
        normal_vector = np.cross(base_vector, np.array([0, 1, 0]))

    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    delta = np.cos(alpha)
    v2 = v1 + delta*base_vector + height*normal_vector
    v3 = v4 + delta*base_vector + height*normal_vector
    points = [v1, v2, v3, v4]
    lengths = [long_base, short_base, side]
    if points is None or lengths is None:
        print('cry')
    else:
        return points, lengths


def create_positions(points, lengths, number_of_beads):
    [v1, v2, v3, v4] = points
    [base14, base23, side] = lengths
    bead_positions = pd.DataFrame(data=np.zeros(
        [number_of_beads+2, 3]), columns=["X", "Y", "Z"], index=np.arange(0, number_of_beads+2))
    generated_positions = np.linspace(v1, v2, int(side)+1)
    for i in range(int(side)+1):
        bead_positions.loc[i, "X"] = generated_positions[i, 0]
        bead_positions.loc[i, "Y"] = generated_positions[i, 1]
        bead_positions.loc[i, "Z"] = generated_positions[i, 2]

    generated_positions = np.linspace(v2, v3, int(base23) + 2)
    for i in range(int(base23)+1):
        bead_positions.loc[int(side) + i, "X"] = generated_positions[i, 0]
        bead_positions.loc[int(side) + i, "Y"] = generated_positions[i, 1]
        bead_positions.loc[int(side) + i, "Z"] = generated_positions[i, 2]

    generated_positions = np.linspace(v3, v4, int(side) + 2)
    for i in range(int(side) + 2):
        bead_positions.loc[int(side) + int(base23) + i,
                          "X"] = generated_positions[i, 0]
        bead_positions.loc[int(side) + int(base23) + i,
                          "Y"] = generated_positions[i, 1]
        bead_positions.loc[int(side) + int(base23) + i,
                          "Z"] = generated_positions[i, 2]


    # for i in bead_positions.index:
    #     coords = bead_positions.iloc[i][["X", "Y", "Z"]]
    #     bead_positions.loc[i, ["X", "Y", "Z"]] = wrap_coords(coords, BOX_SIZE)
    return bead_positions[1:-1]


def init_shape_creation(start, end, number_of_beads, box_size):
    """Generate a chain that forms a trapezoid or triangle."""

    start = wrap_coords(start, BOX_SIZE)
    end = wrap_coords(end, BOX_SIZE)

    raw_diff = np.abs(end - start)
    wrap_diff = np.abs(BOX_SIZE - raw_diff)
    dx, dy, dz = np.minimum(raw_diff, wrap_diff)
    end = np.where(raw_diff < wrap_diff, end, start + wrap_diff)

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    bond_length = 0.97

    end = unwrap_nodes(start, end, box_size)

    if not number_of_beads % 2 and (dist > 0) and (dist <= bond_length):
        points, lengths = upside_down_trapezoid(start, end, number_of_beads, dist)
    else:
        points, lengths = trapezoid(start, end, number_of_beads, dist)
    bead_positions = create_positions(points, lengths, number_of_beads)
    return bead_positions

def update_bond_list(bead_ids, node_id, bond_list):
    """Update bond list with bonds from current chain."""
    bead_ids = np.insert(bead_ids, [0], node_id[0])
    bead_ids = np.append(bead_ids, node_id[1])
    current_list = pd.DataFrame({"BondType": 1, "Atom1": bead_ids[0:-1],
                                 "Atom2": bead_ids[1:]})
    bond_list = pd.concat([bond_list, current_list], ignore_index=True)
    return bond_list


def update_bead_list(bead_data, bead_ids, path, chain):
    """Update bead list with beads frm current chain."""
    path["ID"] = path.index.values + (bead_ids - path.index.values)
    path.set_index("ID", inplace=True)
    bead_data = pd.concat([bead_data, path])
    # bead_data.loc[bead_ids, "X"] = path["X"].copy()
    # bead_data.loc[bead_ids, "Y"] = path["Y"].copy()
    # bead_data.loc[bead_ids, "Z"] = path["Z"].copy()
    bead_data.loc[bead_ids, "Mol"] = chain + 1
    bead_data.loc[bead_ids, "atom-type"] = 2
    return bead_data


def create_chains(full_edge_data, bond_data, bead_data, node_data, box_size, len_of_chain):
    """Create chains."""

    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    chain = -1
    for chain, edge in enumerate(full_edge_data):
        print(chain)
        point_0 = np.asarray([node_data[int(edge[0]), 0], node_data[int(edge[0]), 1],
                   node_data[int(edge[0]), 2]])
        point_n = np.asarray([node_data[int(edge[1]), 0], node_data[int(edge[1]), 1],
                   node_data[int(edge[1]), 2]])
        masterpath = init_shape_creation(
            point_0, point_n, len_of_chain, box_size)

        id_range = np.arange(len(node_data) + (chain * len(masterpath)),
                             len(node_data) + ((chain + 1) * len(masterpath)))
        bead_data = update_bead_list(bead_data, id_range, masterpath, chain)
        bond_data = update_bond_list(id_range, edge, bond_data)

        ax.scatter(masterpath["X"], masterpath["Y"], masterpath["Z"],
                   color=colors[chain], marker='o')
        ax.scatter(point_0[0], point_0[1], point_0[2], color='k', marker='>')
        ax.scatter(point_n[0], point_n[1], point_n[2], color='k', marker='>')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data

# %%
STUDY_NAME = '20241016B1C1'
COORDS = ['x', 'y', 'z']

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = file_functions.load_files(STUDY_NAME, COORDS)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = create_atom_list(NodeData, FullEdges, LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData = create_chains(FullEdges, BondData, BeadData, NodeData, BOX_SIZE, LENGTH_OF_CHAIN)
# BeadData.drop('ID', axis=1, inplace=True)
# BUG: output file is all NANs. Something is not updating or reading correctly
file_functions.write_lammps_data(STUDY_NAME, BeadData, BondData, BOX_SIZE)
# step = 1/(2*np.sqrt(3))
# vertex, distance = upside_down_trapezoid(
#     np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]), 25, 0.5)
# create_positions(vertex, distance, 25)

# step = np.sqrt((1.5**2)/3)
# vertex, distance = trapezoid(
#     np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]), 25, 1.5)
# # create_positions(vertex, distance, 25)
# init_shape_creation([1, 1, 5], np.array([1+step, 1+step, 5+step]), 25, 12)

# step = np.sqrt((5**2)/3)
# vertex, distance = trapezoid(
#     np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]), 25, 5)
# # create_positions(vertex, distance, 25)
# init_shape_creation([1, 1, 5], np.array([1+step, 1+step, 5+step]), 25, 12)

# step = np.sqrt((5**2)/3)
# vertex, distance = trapezoid(
#     np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]), 24, 5)
# # create_positions(vertex, distance, 25)
# init_shape_creation([1, 1, 5], np.array([1+step, 1+step, 5+step]), 24, 12)

# step = np.sqrt((1**2)/3)
# vertex, distance = trapezoid(
#     np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]), 24, 1)
# # create_positions(vertex, distance, 25)
# init_shape_creation([1, 1, 5], np.array([1+step, 1+step, 5+step]), 24, 12)

