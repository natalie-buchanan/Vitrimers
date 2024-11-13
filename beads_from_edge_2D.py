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
    edges = np.loadtxt(os.path.join(base_path, name + '-conn_core_edges.dat'))
    pb_edges = np.loadtxt(os.path.join(base_path, name + '-conn_pb_edges.dat'))
    node_type = np.loadtxt(os.path.join(base_path, name +
                                        '-core_node_type.dat'))
    box_size = np.loadtxt(os.path.join(base_path, name[0:-2] + '-L.dat'))
    # subtract 1 to get number of connecting beads from number of bonds
    len_of_chain = int(np.loadtxt(os.path.join(base_path, name[0:-3] +
                                               '-nu.dat'))[int(name[-1])])
    len_of_chain -= 1
    node_files = np.asarray([core_x, core_y, node_type])
    node_files = node_files.transpose()

    return [node_files, edges, pb_edges, box_size, len_of_chain]


def write_LAMMPS_data(name: str, atom_data, bond_data, box_size,
                      network="auelp"):
    """Write LAMMPS in.data file including positions and bonds."""
    base_path = os.path.join(os.getcwd(), "Data", network)
    data_file = os.path.join(base_path, name + '-in.data')
    with open(data_file, "w",  encoding="utf-8") as file:
        file.write("""{name}

{num_atoms} atoms
{num_bonds} bonds
                   
3 atom types
1 bond types

0 {L_box} xlo xhi
0 {L_box} ylo yhi

Masses\n\n""".format(name=name, num_atoms=len(atom_data),
                   num_bonds=len(bond_data), L_box=box_size))
        for i in atom_data["atom-type"].drop_duplicates().sort_values():
            file.write(f'{int(i)} 1\n')
        file.write("""
Atoms # full\n\n""")
        for i in atom_data.index:
            file.write(f"{int(atom_data.iloc[i]['ID']+1)} " +
                       f"{int(atom_data.iloc[i]['Mol']+1)} " +
                       f"{int(atom_data.iloc[i]['atom-type'])} " +
                       f"{atom_data.iloc[i]['X']} " +
                       f"{atom_data.iloc[i]['Y']} " +
                       "0 \n")
        file.write("""
Bonds\n\n""")
        for i in bond_data.index:
            file.write(f"{i + 1} " +
                       f"{int(bond_data.iloc[i]['BondType'])} " +
                       f"{int(bond_data.iloc[i]['Atom1']) + 1} " +
                       f"{int(bond_data.iloc[i]['Atom2']) + 1} " +
                       "\n")


def create_neighborhood(x, y, factor=0.97, number=9):
    """Create neighborhood."""
    points = []
    # Adjust number for finer granularity
    for theta in np.linspace(0, 2*np.pi, number):
            xn = x + factor * np.cos(theta)
            yn = y + factor * np.sin(theta)

            if all(v == 0 for v in [xn, yn]):
                print("Here")
            points.append([xn, yn])
    points = np.array(points)

    return np.array(points)


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
    while np.any(np.less(z1, 0)):
        z1[z1 < 0] += L
    while np.any(np.greater(z1, L)):
        z1[z1 > L] -= L
    return z1


def check_distances(A, B, distance, box_size):
    """
    Checks if any row in array A is within a given distance of any row in array B,
    considering periodic boundary conditions.

    Args:
        A: A NumPy array of shape (N, 3) representing the coordinates of points.
        B: A NumPy array of shape (M, 3) representing the coordinates of other points.
        distance: The maximum distance for a point to be considered "close".
        box_size: The size of the periodic box in each dimension (assumed to be cubic).

    Returns:
        A boolean array of shape (N,) indicating whether each row in A is close to any row in B.
    """

    num_points_A = A.shape[0]
    is_close = np.zeros(num_points_A, dtype=bool)

    for i in range(num_points_A):
        for j in range(B.shape[0]):
            dx = np.abs(A[i, 0] - B[j, 0])
            dy = np.abs(A[i, 1] - B[j, 1])

            # Periodic boundary conditions
            dx = np.minimum(dx, box_size - dx)
            dy = np.minimum(dy, box_size - dy)

            dist = np.sqrt(dx**2 + dy**2)

            if dist < distance:
                is_close[i] = True
                break  # Move to the next point in A if a close point is found
    return is_close


def check_neighborhood(neighborhood, bead_positions, cutoff=0.25):
    """Check to see if neighbors are occupied."""
    truth = check_distances(neighborhood, bead_positions, cutoff, BOX_SIZE)
    test = neighborhood[~truth]
    if not test.any():
        return False
    return test


def find_neighborhood(x, y, bead_positions):
    """Create neighborhood that is not fully occupied."""
    neighborhood = create_neighborhood(x, y)
    check = check_neighborhood(neighborhood, bead_positions)
    if not isinstance(check, np.ndarray):
        print('Here')
        # generate neighborhood of points 1.5 away if all 1 away are occupied
        numbertries = [3, 4, 5, 11, 13, 16, 20]
        while len(numbertries) > 0:
            num_index = random.randint(0, len(numbertries)-1)
            interval = numbertries.pop(num_index)
            neighborhood = create_neighborhood(x, y, number=interval)
            if not isinstance(check_neighborhood(neighborhood, bead_positions),
                            np.ndarray):
                raise ValueError(
                    'Neighborhood full: ' + str(neighborhood))
    return neighborhood


def select_indices(theta, neighborhood, current_point):
    """Select indices based on theta and neighborhood."""
    random.seed(42)
    valid_indices = np.where(np.any(neighborhood != 0, axis=1))[0]
    indices = np.empty([len(valid_indices), 2])
    for i in range(neighborhood.shape[1]):
        column = neighborhood[valid_indices, 1]
        if random.random() < theta[i]:
            indices[:, i] = column < current_point[i]
        else:
            indices[:, i] = column >= current_point[i]
    return indices


def calculate_theta(current_point, target_point, n, i):
    """Calculate chance of moving in positive or negative direction."""
    x, y = current_point
    xtarg, ytarg= target_point
    denom = (0.97 / 2.0) * (n - (i * 2))

    theta_x = 0.5 * (1 - ((xtarg - x) / denom))
    theta_y = 0.5 * (1 - ((ytarg - y) / denom))
    return theta_x, theta_y


def step_choice(i, current_point, target_point, bead_positions, n):
    """Choose next step in constrained walk."""
    x, y = current_point

    theta = calculate_theta(current_point, target_point, n, i)

    region = find_neighborhood(x, y, bead_positions)
    rows = np.any(region != 0, axis=1)
    region = region[rows]
    index = select_indices(theta, region, current_point)

    row_sums = np.sum(index, axis=1)
    max_sum = np.max(row_sums)
    options = np.where(row_sums == max_sum)[0]
    choice = np.random.choice(options)
    if not np.any(region[choice]):
        print(choice)
    return region[choice]


def constrained_walk(start, end, n=15):
    """Generate atom position in a constrained walk between two nodes."""
    # [x0, xn] = wrap_coords([start[0], end[0]], BOX_SIZE)
    # [y0, yn] = wrap_coords([start[1], end[1]], BOX_SIZE)
    [x0, y0] = start
    [xn, yn] = end
    [x0, y0] = unwrap_coords(start, end, BOX_SIZE)
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
        # current_point = unwrap_coords(current_point, target_point, BOX_SIZE)
        x, y = step_choice(i, current_point, target_point,
                              bead_positions, n)
        bead_positions[i + 1] = [x, y]
        # bead_positions[i+1] = wrap_coords([x, y], BOX_SIZE)
        # Step from ending side
        current_point = xn, yn
        target_point = x, y
        xn, yn = step_choice(i, current_point, target_point,
                                 bead_positions, n)
        bead_positions[n - (i + 1)] = [xn, yn]
        # bead_positions[n + 1 - i] = wrap_coords([xn, yn], BOX_SIZE)
    return bead_positions[1:-1]


def calculate_wrapped_distance(array):
    """sdsd."""
    num_rows = array.shape[0]
    results = np.zeros_like(array)

    for i in range(0, num_rows - 1, 1):
        new_row1 = unwrap_coords(array[i], array[i + 1], BOX_SIZE)
        # Calculate the difference between new_row1 and row i+1
        diff = abs(new_row1 - array[i + 1])
        results[i] = diff

    return np.array(results)


def calculate_wrapped_distance_full(array):
    """Calculate actual distance, not difference."""
    num_rows = array.shape[0]
    results = np.zeros((num_rows, 1))
    for i in range(0, num_rows - 1, 1):
        new_row = unwrap_coords(array[i], array[i+1], BOX_SIZE)
        results[i] = math.sqrt((new_row[0]-array[i + 1, 0])**2 +
                               (new_row[1] - array[i + 1, 1])**2)
    return results


def create_atom_list(node_data, edge_data, length_of_chain):
    """Create panda Data Frame of generated positions and nodes."""
    x_data, y_data, node_type = node_data[:,0], node_data[:, 1], node_data[:, 2]
    atom_list = pd.DataFrame(data={'ID':
                                   np.arange(0, ((length_of_chain) *
                                                 len(edge_data) +
                                                 len(x_data)), 1),
                                   'X': np.nan, 'Y': np.nan, 'Mol': np.nan})
    atom_list.loc[np.arange(0, len(x_data)), "X"] = x_data
    atom_list.loc[np.arange(0, len(x_data)), "Y"] = y_data
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
    [node_x, node_y, node_type] = [node_data[:, 0], node_data[:, 1], node_data[:, 2]]
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    plt.figure()
    fullcycle = pd.DataFrame(data=np.zeros([len(full_edge_data), 2]),
                             columns=["Cycles", "Max Dist"])

    for chain, edge in enumerate(full_edge_data):
        print(chain)
        point_0 = (node_x[int(edge[0])], node_y[int(edge[0])])
        point_n = (node_x[int(edge[1])], node_y[int(edge[1])])
        maxdist = 50  # arbitrary large value to start
        cycle = 0
        while maxdist > 1 and cycle < 25:  # arbitrary cut offs
            path = constrained_walk(start=point_0, end=point_n,
                                    n=LENGTH_OF_CHAIN)
            curr = max(calculate_wrapped_distance_full(path))[0]
            cycle += 1
            if curr < maxdist:
                maxdist = curr
                masterpath = path
            maxdist = min(maxdist, curr)
        for i in range(len(masterpath)):
            masterpath[i] = wrap_coords(masterpath[i], BOX_SIZE)
        path_x = masterpath[:, 0]
        path_y = masterpath[:, 1]
        fullcycle.iloc[chain] = [cycle, maxdist]
        plt.scatter(path_x, path_y, color=colors[chain], marker='o')
        # ax.plot(path_x, path_y, path_z, color=colors[chain], linestyle='-')
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
            raise ValueError('Error in chain ' + str(chain)
                             + " Max Dist: " + str(maxdist))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data, fullcycle


# %%
STUDY_NAME = '20241016B1C1'

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = create_atom_list(NodeData, FullEdges, LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData, runInfo = create_chains(FullEdges, BondData, BeadData,
                                            NodeData)
write_LAMMPS_data(STUDY_NAME, BeadData, BondData, BOX_SIZE)

# %% PLot
colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(BeadData)))
plt.figure()
# ax.scatter(BeadData[BeadData['Mol'] == 0]["X"],
#            BeadData[BeadData['Mol'] == 0]["Y"],
#            BeadData[BeadData['Mol'] == 0]["Z"],
#            marker='>', color='k')
plt.scatter(BeadData[BeadData['Mol'] == 10]["X"],
           BeadData[BeadData['Mol'] == 10]["Y"],
           marker='.', c=colors[BeadData['Mol'] == 10], alpha=0.7)
plt.show()


colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(BeadData)))
plt.figure()
plt.scatter(BeadData[BeadData['Mol'] == 0]["X"],
           BeadData[BeadData['Mol'] == 0]["Y"],
           marker='>', color='k')
plt.plot(BeadData[BeadData['Mol'] == 10]["X"],
        BeadData[BeadData['Mol'] == 10]["Y"],
        marker='.')
plt.title('10')
plt.show()


# %%
colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(BeadData)))
node_x = NodeData[:, 0]
node_y = NodeData[:, 1]
Node = [node_x[int(FullEdges[-1][0])], node_y[int(FullEdges[-1][0])]]
Node2 = [node_x[int(FullEdges[-1][1])], node_y[int(FullEdges[-1][1])]]
Node2 = unwrap_coords(Node2, Node, BOX_SIZE)
tester = BeadData[BeadData['Mol'] == 10][["X", "Y"]]
tester = tester.apply(unwrap_coords, axis=1, result_type='broadcast', second_point=Node, L=BOX_SIZE)
plt.figure()
plt.plot(tester["X"],tester["Y"],marker='.')
plt.scatter(Node[0], Node[1], c='k', marker='o')
plt.scatter(Node2[0], Node2[1], c='k', marker='*')
plt.title('Unwrapped')
plt.show()