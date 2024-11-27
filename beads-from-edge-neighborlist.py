# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:45:37 2024

@author: Natalie.Buchanan
"""

import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import file_functions


def unwrap_coords(first_point, second_point, L):
    """Unwrap pair of coordinates."""
    v1 = np.array(first_point)
    v2 = np.array(second_point)
    diff = abs(v1 - v2)
    z1 = v1.copy()
    # wrap_flag = np.zeros_like(v1)
    # wrap_flag[(diff > L/2) & (v1 < v2)] = 1
    # wrap_flag[(diff > L/2) & (v1 > v2)] = -1
    z1[(diff > L/2) & (v1 < v2)] += L
    z1[(diff > L/2) & (v1 > v2)] -= L

    return z1


def wrap_coords(first_point, L):
    """Wrap pair of coordinates."""
    v1 = np.array(first_point, dtype='float64')
    v1 = v1 % L
    # z1 = v1.copy()
    # z1[v1 > L] -= L
    # z1[v1 < 0] += L
    return v1


def calculate_wrapped_distance(array):
    """sdsd."""
    num_rows = array.shape[0]
    results = np.zeros_like(array)

    for i in range(0, num_rows - 1, 1):
        new_row1 = unwrap_coords(array[i], array[i + 1], BOX_SIZE)

        # Calculate the difference between new_row1 and row i+1
        diff = new_row1 - array[i + 1]
        results[i] = diff

    return np.array(results)


def create_neighborhood(x, y, factor=0.97):
    """Create neighborhood."""
    return [(x-factor, y-factor), (x - factor, y), (x - factor, y + factor),
            (x, y - factor), (False, False), (x, y + factor),
            (x + factor, y - factor), (x + factor, y),
            (x + factor, y + factor)]


def check_neighborhood(neighborhood, bead_positions, cutoff=0.1):
    """Check to see if neighbors are occupied."""
    for j, pair in enumerate(neighborhood):
        wrapped_positions = np.apply_along_axis(unwrap_coords, 1,
                                                bead_positions,
                                                second_point=pair,
                                                L=BOX_SIZE)
        checkx = np.where(abs(wrapped_positions[:, 0] - pair[0]) <= cutoff)[0]
        checky = np.where(abs(wrapped_positions[:, 1] - pair[1]) <= cutoff)[0]
        if np.isin(checkx, checky).any():
            neighborhood[j] = (False, False)
    # If all occupied, returns False, else returns neighborhood
    if all(t == (False, False) for t in neighborhood):
        return False
    return neighborhood


def find_neighborhood(x, y, bead_positions):
    """Create neighborhood that is not fully occupied."""
    neighborhood = create_neighborhood(x, y, BOND_LENGTH)
    if not check_neighborhood(neighborhood, bead_positions):
        # generate neighborhood of points 1.5 away if all 1 away are occupied
        neighborhood = create_neighborhood(x, y, factor=.7*BOND_LENGTH)
        if not check_neighborhood(neighborhood, bead_positions):
            raise ValueError(
                'Neighborhood full: ' +
                str(check_neighborhood(neighborhood, bead_positions)))
    return neighborhood


def select_indices(theta, neighborhood, is_x_axis):
    """Select indices based on theta and neighborhood."""
    if is_x_axis:
        if random.random() <= theta:
            indices = (0, 1, 2)
        else:
            indices = (6, 7, 8)
    else:  # y data instead
        if random.random() <= theta:
            indices = (0, 3, 6)
        else:
            indices = (2, 5, 8)
    pos = [neighborhood[i] for i in indices]
    if all(t == (False, False) for t in pos):
        if is_x_axis:
            indices = (3, 4, 5)
        else:
            indices = (1, 4, 7)
        pos = [neighborhood[i] for i in indices]
        if all(t == (False, False) for t in pos):
            indices = tuple(i for i, tup in enumerate(neighborhood)
                            if tup != (False, False))
    return indices


def calculate_theta(current_point, target_point, n, i):
    """Calculate chance of moving in positive or negative direction."""
    x, y = current_point
    xtarg, ytarg = target_point
    if (n - (i * 2)) == 0:
        denom = 1*BOND_LENGTH
    else:
        denom = (n - (i * 2))*BOND_LENGTH
    theta_x = 0.5 * (1 - ((xtarg - x) / denom))
    theta_y = 0.5 * (1 - ((ytarg - y) / denom))
    return theta_x, theta_y


def step_choice(i, current_point, target_point, bead_positions, n):
    """Choose next step in constrained walk."""
    x, y = current_point

    theta_x, theta_y = calculate_theta(current_point, target_point, n, i)

    neighborhood = find_neighborhood(x, y, bead_positions)
    index_x = select_indices(theta_x, neighborhood, True)
    index_y = select_indices(theta_y, neighborhood, False)

    options = (tuple(set(index_x) & set(index_y)))
    options = [elem for elem in options if all(neighborhood[elem])]
    if options:
        choice = random.choice(options)
    else:
        options = [elem for elem in index_x + index_y if
                   all(neighborhood[elem])]
        if options:
            choice = random.choice(options)
        else:
            raise ValueError("No options available")
    return neighborhood[choice]


def constrained_walk(start, end, n):
    """Generate atom position in a constrained walk between two nodes."""
    [x0, xn] = wrap_coords([start[0], end[0]], BOX_SIZE)
    [y0, yn] = wrap_coords([start[1], end[1]], BOX_SIZE)
    x = x0
    y = y0
    bead_positions = np.empty((n+2, 2))
    bead_positions[:] = np.nan
    bead_positions[0] = [x0, y0]
    bead_positions[-1] = [xn, yn]

    for i in range(int((n+2)/2)):
        # Step from starting side
        current_point = x, y
        target_point = xn, yn
        current_point = unwrap_coords(current_point, target_point, BOX_SIZE)
        x, y = step_choice(i, current_point, target_point, bead_positions, n)
        bead_positions[i+1] = wrap_coords([x, y], BOX_SIZE)
        # Step from ending side
        current_point = xn, yn
        target_point = x, y
        xn, yn = step_choice(i, current_point, target_point, bead_positions, n)
        bead_positions[n + 1 - i] = wrap_coords([xn, yn], BOX_SIZE)
    return bead_positions[1:-1]


def update_bond_list(bead_ids, node_id, bond_list):
    """Update bond list with bonds from current chain."""
    bead_ids = np.insert(bead_ids, [0], node_id[0])
    bead_ids = np.append(bead_ids, node_id[1])
    current_list = pd.DataFrame({"BondType": 1, "Atom1": bead_ids[0:-1],
                                 "Atom2": bead_ids[1:]})
    bond_list = pd.concat([bond_list, current_list], ignore_index=True)
    return bond_list


def create_chains(full_edge_data, bond_data, bead_data, node_data):
    """Create chains."""
    [node_x, node_y, node_type] = node_data
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    plt.figure()
    fullcycle = np.zeros([len(full_edge_data), 2])
    for chain, edge in enumerate(full_edge_data):
        point_0 = (node_x[int(edge[0])], node_y[int(edge[0])])
        point_n = (node_x[int(edge[1])], node_y[int(edge[1])])
        maxdist = 50  # arbitrary large value to start
        cycle = 0
        while maxdist > 3 and cycle < 25:  # arbitrary cut offs
            path = constrained_walk(start=point_0, end=point_n,
                                    n=LENGTH_OF_CHAIN)
            path_x = path[:, 0]
            path_y = path[:, 1]
            curr = max([max(calculate_wrapped_distance(path_x)),
                        max(calculate_wrapped_distance(path_y))])
            cycle += 1
            maxdist = min(maxdist, curr)
        fullcycle[chain] = [cycle, maxdist]
        plt.scatter(path_x, path_y, color=colors[chain], marker='o')
        # plt.plot(path_x, path_y, color=colors[chain], linestyle='-')
        plt.scatter(point_0[0], point_0[1], color='k', marker='>')
        plt.scatter(point_n[0], point_n[1], color='k', marker='>')
        id_range = np.arange(len(node_x) + (chain * len(path_x)),
                             len(node_x) + ((chain + 1) * len(path_x)))
        bead_data.loc[id_range, "X"] = path_x
        bead_data.loc[id_range, "Y"] = path_y
        bead_data.loc[id_range, "Mol"] = chain + 1
        bead_data.loc[id_range, "atom-type"] = 2
        bond_data = update_bond_list(id_range, edge, bond_data)

        if maxdist > 3:
            plt.gca().set_aspect('equal')
            plt.show()
            print(max(calculate_wrapped_distance(path_x)),
                  max(calculate_wrapped_distance(path_y)))
            raise ValueError('Error in chain ' + str(chain))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data


# %% Create Network

STUDY_NAME = '20241016A1C1'
BOND_LENGTH = .97

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = \
    file_functions.load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = file_functions.create_atom_list(NodeData, FullEdges,
                                           LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData = create_chains(FullEdges, BondData, BeadData, NodeData)
file_functions.write_LAMMPS_data(STUDY_NAME, BeadData, BondData, BOX_SIZE)

# %% Check
# test = BeadData.copy()
# for i in range(len(test)-2):
#     c_point = BeadData.loc[i, ["X", "Y"]]
#     x_point = BeadData.loc[i, ["X", "Y"]]
#     n_point = unwrap_coords(c_point, x_point, BOX_SIZE)
#     test.loc[i, ["X", "Y"]] = n_point

# plt.figure()
# nodes = test[test["Mol"] == 0]
# other = test[test["Mol"] != 0]
# color = mpl.colormaps['rainbow'](np.linspace(0, 1, int(max(test["Mol"]))))
# plt.scatter(nodes["X"], nodes["Y"], c='k', marker='>')
# plt.scatter(other["X"], other["Y"], c=other["Mol"])
# plt.show()
