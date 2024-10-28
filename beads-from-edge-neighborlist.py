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


def load_files(name: str, network="auelp"):
    """Load data from files core_x, core_y, and edges."""
    base_path = os.path.join(os.getcwd(), "Data", network)
    core_x = np.loadtxt(os.path.join(base_path, name + '-core_x.dat'))
    core_y = np.loadtxt(os.path.join(base_path, name + '-core_y.dat'))
    edges = np.loadtxt(os.path.join(base_path, name + '-conn_core_edges.dat'))
    pb_edges = np.loadtxt(os.path.join(base_path, name + '-conn_pb_edges.dat'))
    node_type = np.loadtxt(os.path.join(base_path, name +
                                        '-core_node_type.dat'))
    box_size = np.loadtxt(os.path.join(base_path, name[0:-2] + '-L.dat'))
    return [core_x, core_y, node_type, edges, pb_edges, box_size]


def create_neighborhood(x, y, factor=1):
    """Create neighborhood."""
    return [(x-factor, y-factor), (x - factor, y), (x - factor, y + factor),
            (x, y - factor), (False, False), (x, y + factor),
            (x + factor, y - factor), (x + factor, y),
            (x + factor, y + factor)]


def unwrap_coords(first_point, second_point, L):
    """Unwrap pair of coordinates."""
    v1 = np.array(first_point)
    v2 = np.array(second_point)
    diff = abs(v1 - v2)
    z1 = v1.copy()
    wrap_flag = np.zeros_like(v1)
    wrap_flag[(diff > L/2) & (v1 < v2)] = 1
    wrap_flag[(diff > L/2) & (v1 > v2)] = -1
    z1[(diff > L/2) & (v1 < v2)] += L
    z1[(diff > L/2) & (v1 > v2)] -= L

    return z1


def wrap_coords(first_point, L):
    """Wrap pair of coordinates."""
    v1 = np.array(first_point, dtype='float64')
    z1 = v1.copy()
    z1[v1 > L] -= L
    z1[v1 < 0] += L
    return z1


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
    neighborhood = create_neighborhood(x, y)
    if not check_neighborhood(neighborhood, bead_positions):
        # generate neighborhood of points 1.5 away if all 1 away are occupied
        neighborhood = create_neighborhood(x, y, factor=1.5)
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

    theta_x = 0.5 * (1 - ((xtarg - x) / (n - (i * 2))))
    theta_y = 0.5 * (1 - ((ytarg - y) / (n - (i * 2))))
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


def constrained_walk(start, end, n=15):
    """Generate atom position in a constrained walk between two nodes."""
    [x0, xn] = wrap_coords([start[0], end[0]], BOX_SIZE)
    [y0, yn] = wrap_coords([start[1], end[1]], BOX_SIZE)
    x = x0
    y = y0
    bead_positions = np.empty((n+1, 2))
    bead_positions[:] = np.nan
    bead_positions[0] = [x0, y0]
    bead_positions[-1] = [xn, yn]

    for i in range(int(n/2)):
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
        bead_positions[n-i-1] = wrap_coords([xn, yn], BOX_SIZE)
    return bead_positions[1:-1]


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


def create_atom_list(x_data, y_data, node_type, edge_data):
    """Create panda Data Frame of generated positions and nodes."""
    atom_list = pd.DataFrame(data=np.arange(0, ((LENGTH_OF_CHAIN + 1)
                                                * len(edge_data) +
                                                len(x_data)), 1),
                             columns=['ID'])
    atom_list['X'] = np.nan
    atom_list['Y'] = np.nan
    atom_list['Mol'] = np.nan
    atom_list.loc[np.arange(0, len(NodeX)), "X"] = x_data
    atom_list.loc[np.arange(0, len(NodeX)), "Y"] = y_data
    atom_list.loc[np.arange(0, len(NodeX)), "atom-type"] = node_type
    atom_list.loc[np.arange(0, len(NodeX)), "Mol"] = 0
    return atom_list


STUDY_NAME = '20241016A1C0'
LENGTH_OF_CHAIN = 25

[NodeX, NodeY, NodeType, Edges, PB_edges, BOX_SIZE] = load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
bead_data = create_atom_list(NodeX, NodeY, NodeType, FullEdges)


# %% Plot All Edges
cmap = mpl.colormaps['rainbow']
colors = cmap(np.linspace(0, 1, len(FullEdges)))
plt.figure()
counter = 0
fullcycle = np.zeros([len(FullEdges), 2])
for chain, edge in enumerate(FullEdges):
    point0 = (NodeX[int(edge[0])], NodeY[int(edge[0])])
    pointN = (NodeX[int(edge[1])], NodeY[int(edge[1])])
    maxdist = 50
    cycle = 0
    while maxdist > 3 and cycle < 25:  # arbitrary cut offs
        path = constrained_walk(start=point0, end=pointN,
                                n=LENGTH_OF_CHAIN)
        pathX = path[:, 0]
        pathY = path[:, 1]
        curr = max([max(calculate_wrapped_distance(pathX)),
                    max(calculate_wrapped_distance(pathY))])
        cycle += 1
        maxdist = min(maxdist, curr)
    fullcycle[chain] = [cycle, maxdist]
    plt.scatter(pathX, pathY, color=colors[counter], marker='o')
    # plt.plot(pathX, pathY, color=colors[counter], linestyle='-')
    plt.scatter(point0[0], point0[1], color='k', marker='>')
    plt.scatter(pointN[0], pointN[1], color='k', marker='>')
    counter += 1
    bead_data.loc[np.arange(len(NodeX) + (chain * len(pathX)),
                            len(NodeX) + ((chain + 1) * len(pathX))),
                  "X"] = pathX
    bead_data.loc[np.arange(len(NodeY) + (chain * len(pathY)),
                            len(NodeY) + ((chain + 1) * len(pathY))),
                  "Y"] = pathY
    bead_data.loc[np.arange(len(NodeX) + (chain * len(pathX)),
                            len(NodeX) + ((chain + 1) * len(pathX))),
                  "Mol"] = chain + 1
    bead_data.loc[np.arange(len(NodeX) + (chain * len(pathX)),
                            len(NodeX) + ((chain + 1) * len(pathX))),
                  "atom-type"] = 2
    if maxdist > 3:
        plt.gca().set_aspect('equal')
        plt.show()
        print(max(calculate_wrapped_distance(pathX)),
              max(calculate_wrapped_distance(pathY)))
        raise ValueError('Error in chain ' + str(chain))
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Path")
plt.gca().set_aspect('equal')
plt.show()
