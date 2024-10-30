# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:37:25 2024

@author: Natalie.Buchanan
"""

import random
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_files(name: str, network="auelp"):
    """Load data from files core_x, core_y, and edges."""
    base_path = os.path.join(os.getcwd(), "Data", network)
    core_x = np.loadtxt(os.path.join(base_path, name + '-core_x.dat'))
    core_y = np.loadtxt(os.path.join(base_path, name + '-core_y.dat'))
    core_z = np.loadtxt(os.path.join(base_path, name + '-core_z.dat'))
    edges = np.loadtxt(os.path.join(base_path, name + '-conn_core_edges.dat'))
    pb_edges = np.loadtxt(os.path.join(base_path, name + '-conn_pb_edges.dat'))
    node_type = np.loadtxt(os.path.join(base_path, name +
                                        '-core_node_type.dat'))
    box_size = np.loadtxt(os.path.join(base_path, name[0:-2] + '-L.dat'))
    node_files = [core_x, core_y, core_z, node_type]

    return [node_files, edges, pb_edges, box_size]


def create_neighborhood(x, y, z, factor=1):
    """Create neighborhood."""
    return [(x - factor, y - factor, z - factor),
            (x - factor, y, z - factor),
            (x - factor, y + factor, z - factor),
            (x, y - factor, z - factor), (x, y, z - factor),
            (x, y + factor, z - factor), (x + factor, y - factor, z - factor),
            (x + factor, y, z - factor), (x + factor, y + factor, z - factor),
            (x - factor, y - factor, z), (x - factor, y, z),
            (x - factor, y + factor, z),
            (x, y - factor, z), False,
            (x, y + factor, z), (x + factor, y - factor, z),
            (x + factor, y, z), (x + factor, y + factor, z),
            (x - factor, y - factor, z + factor), (x - factor, y, z + factor),
            (x - factor, y + factor, z + factor),
            (x, y - factor, z + factor), (x, y, z + factor),
            (x, y + factor, z + factor), (x + factor, y - factor, z + factor),
            (x + factor, y, z + factor), (x + factor, y + factor, z + factor)]


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


def check_neighborhood(neighborhood, bead_positions, cutoff=0):
    """Check to see if neighbors are occupied."""
    for j, pair in enumerate(neighborhood):
        if not pair:
            neighborhood[j] = False
        else:
            wrapped_positions = np.apply_along_axis(unwrap_coords, 1,
                                                    bead_positions,
                                                    second_point=pair,
                                                    L=BOX_SIZE)
            checkx = np.where(abs(wrapped_positions[:, 0] - pair[0]) <= cutoff)[0]
            checky = np.where(abs(wrapped_positions[:, 1] - pair[1]) <= cutoff)[0]
            checkz = np.where(abs(wrapped_positions[:, 2] - pair[2]) <= cutoff)[0]
    
            if np.isin(checkx, checky, checkz).any():
                neighborhood[j] = False
    # If all occupied, returns False, else returns neighborhood
    if not any(t for t in neighborhood):
        return False
    return neighborhood


def find_neighborhood(x, y, z, bead_positions):
    """Create neighborhood that is not fully occupied."""
    neighborhood = create_neighborhood(x, y, z)
    if not check_neighborhood(neighborhood, bead_positions):
        # generate neighborhood of points 1.5 away if all 1 away are occupied
        neighborhood = create_neighborhood(x, y, z, factor=.5)
        if not check_neighborhood(neighborhood, bead_positions):
            raise ValueError(
                'Neighborhood full: ' + str(neighborhood))
    return neighborhood


def select_indices(theta, neighborhood, axis):
    """Select indices based on theta and neighborhood."""
    range_list = range(0, 27)
    if axis == 'x':
        if random.random() <= theta:
            indices = [range_list[i] for j in range(0, len(range_list), 9)
                       for i in range(j, j+3) if i < len(range_list)]
        else:
            indices = [range_list[i] for j in range(6, len(range_list), 9)
                       for i in range(j, j+3) if i < len(range_list)]
    elif axis == 'y':  # y data instead
        if random.random() <= theta:
            indices = list(range(0, 27, 3))
        else:
            indices = list(range(2, 27, 3))
    elif axis == 'z':
        if random.random() <= theta:
            indices = list(range(0, 9))
        else:
            indices = list(range(18, 27))
    else:
        raise ValueError("Invalid axis command.")

    pos = [neighborhood[i] for i in indices]
    if not any(t for t in pos):
        if axis == 'x':
            indices = [range_list[i] for j in range(3, len(range_list), 9)
                       for i in range(j, j+3) if i < len(range_list)]
        elif axis == 'y':
            indices = list(range(0, 27, 3))
        elif axis == 'z':
            indices = list(range(9, 17))
        else:
            raise ValueError("Invalid axis command.")

        pos = [neighborhood[i] for i in indices]
        if not any(t for t in pos):
            indices = tuple(i for i, tup in enumerate(neighborhood)
                            if tup != (False, False))
    return indices


def calculate_theta(current_point, target_point, n, i):
    """Calculate chance of moving in positive or negative direction."""
    x, y, z = current_point
    xtarg, ytarg, ztarg = target_point

    theta_x = 0.5 * (1 - ((xtarg - x) / (n - (i * 2))))
    theta_y = 0.5 * (1 - ((ytarg - y) / (n - (i * 2))))
    theta_z = 0.5 * (1 - ((ztarg - z) / (n - (i * 2))))
    return theta_x, theta_y, theta_z


def step_choice(i, current_point, target_point, bead_positions, n):
    """Choose next step in constrained walk."""
    x, y, z = current_point

    theta_x, theta_y, theta_z = calculate_theta(current_point,
                                                target_point, n, i)

    neighborhood = find_neighborhood(x, y, z, bead_positions)
    index_x = select_indices(theta_x, neighborhood, 'x')
    index_y = select_indices(theta_y, neighborhood, 'y')
    index_z = select_indices(theta_z, neighborhood, 'z')

    options = (tuple(set(index_x) & set(index_y) & set(index_z)))
    options = [elem for elem in options if
               isinstance(neighborhood[elem], (list, tuple))
               and all(neighborhood[elem])]
    if options:
        choice = random.choice(options)
    else:
        options = [elem for elem in index_x + index_y + index_z if
                   isinstance(neighborhood[elem], (list, tuple))
                   and all(neighborhood[elem])]
        if options:
            choice = random.choice(options)
        else:
            raise ValueError("No options available")
    return neighborhood[choice]


def constrained_walk(start, end, n=15):
    """Generate atom position in a constrained walk between two nodes."""
    [x0, xn] = wrap_coords([start[0], end[0]], BOX_SIZE)
    [y0, yn] = wrap_coords([start[1], end[1]], BOX_SIZE)
    [z0, zn] = wrap_coords([start[2], end[2]], BOX_SIZE)
    x = x0
    y = y0
    z = z0
    bead_positions = np.empty((n+1, 3))
    bead_positions[:] = np.nan
    bead_positions[0] = [x0, y0, z0]
    bead_positions[-1] = [xn, yn, zn]

    for i in range(int(n/2)):
        # Step from starting side
        current_point = x, y, z
        target_point = xn, yn, zn
        current_point = unwrap_coords(current_point, target_point, BOX_SIZE)
        x, y, z = step_choice(i, current_point, target_point,
                              bead_positions, n)
        bead_positions[i+1] = wrap_coords([x, y, z], BOX_SIZE)
        # Step from ending side
        current_point = xn, yn, zn
        target_point = x, y, zn
        xn, yn, zn = step_choice(i, current_point, target_point,
                                 bead_positions, n)
        bead_positions[n-i-1] = wrap_coords([xn, yn, zn], BOX_SIZE)
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


def calculate_wrapped_distance_full(array):
    """Calculate actual distance, not difference."""
    num_rows = array.shape[0]
    results = np.zeros((num_rows, 1))
    for i in range(0, num_rows - 1, 1):
        new_row = unwrap_coords(array[i], array[i+1], BOX_SIZE)
        results[i] = math.sqrt((new_row[0]-array[i + 1, 0])**2 +
                               (new_row[1] - array[i + 1, 1])**2 +
                               (new_row[2] - array[i + 1, 2])**2)
    return results


def create_atom_list(node_data, edge_data):
    """Create panda Data Frame of generated positions and nodes."""
    x_data, y_data, z_data, node_type = node_data
    atom_list = pd.DataFrame(data={'ID':
                                   np.arange(0, ((LENGTH_OF_CHAIN) *
                                                 len(edge_data) +
                                                 len(x_data)), 1),
                                   'X': np.nan, 'Y': np.nan, 'Z': np.nan,
                                   'Mol': np.nan})
    atom_list.loc[np.arange(0, len(x_data)), "X"] = x_data
    atom_list.loc[np.arange(0, len(x_data)), "Y"] = y_data
    atom_list.loc[np.arange(0, len(x_data)), "Z"] = z_data
    atom_list.loc[np.arange(0, len(x_data)), "atom-type"] = node_type
    atom_list.loc[np.arange(0, len(x_data)), "Mol"] = 0
    return atom_list


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
    [node_x, node_y, node_z, node_type] = node_data
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fullcycle = np.zeros([len(full_edge_data), 2])
    for chain, edge in enumerate(full_edge_data):
        point_0 = (node_x[int(edge[0])], node_y[int(edge[0])],
                   node_z[int(edge[0])])
        point_n = (node_x[int(edge[1])], node_y[int(edge[1])],
                   node_z[int(edge[1])])
        maxdist = 50  # arbitrary large value to start
        cycle = 0
        while maxdist > 3 and cycle < 25:  # arbitrary cut offs
            path = constrained_walk(start=point_0, end=point_n,
                                    n=LENGTH_OF_CHAIN)
            path_x = path[:, 0]
            path_y = path[:, 1]
            path_z = path[:, 2]
            curr = max(calculate_wrapped_distance_full(path))[0]
            cycle += 1
            maxdist = min(maxdist, curr)
        fullcycle[chain] = [cycle, maxdist]
        ax.scatter(path_x, path_y, path_z, color=colors[chain], marker='o')
        # ax.plot(path_x, path_y, path_z, color=colors[chain], linestyle='-')
        plt.scatter(point_0[0], point_0[1], point_0[2], color='k', marker='>')
        plt.scatter(point_n[0], point_n[1], point_n[2], color='k', marker='>')
        id_range = np.arange(len(node_x) + (chain * len(path_x)),
                             len(node_x) + ((chain + 1) * len(path_x)))
        bead_data.loc[id_range, "X"] = path_x
        bead_data.loc[id_range, "Y"] = path_y
        bead_data.loc[id_range, "Z"] = path_z
        bead_data.loc[id_range, "Mol"] = chain + 1
        bead_data.loc[id_range, "atom-type"] = 2
        bond_data = update_bond_list(id_range, edge, bond_data)

        if maxdist > 3:
            plt.gca().set_aspect('equal')
            plt.show()
            print(max(calculate_wrapped_distance(path_x)),
                  max(calculate_wrapped_distance(path_y)),
                  max(calculate_wrapped_distance(path_z)))
            raise ValueError('Error in chain ' + str(chain))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data, fullcycle


# %%
STUDY_NAME = '20241016B1C1'
LENGTH_OF_CHAIN = 25

[NodeData, Edges, PB_edges, BOX_SIZE] = load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = create_atom_list(NodeData, FullEdges)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData, runInfo = create_chains(FullEdges, BondData, BeadData,
                                            NodeData)

# %% PLot
colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(BeadData)))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(BeadData[BeadData['Mol'] == 0]["X"],
           BeadData[BeadData['Mol'] == 0]["Y"],
           BeadData[BeadData['Mol'] == 0]["Z"],
           marker='>', color='k')
ax.scatter(BeadData[BeadData['Mol'] == 10]["X"],
           BeadData[BeadData['Mol'] == 10]["Y"],
           BeadData[BeadData['Mol'] == 10]["Z"],
           marker='.', c=colors[BeadData['Mol'] == 10], alpha=0.7)

plt.show()
