"""
Created on Mon Oct 28 13:37:25 2024

@author: Natalie.Buchanan
"""

import random
import math
import multiprocessing
import pandas as pd
import numpy as np
from tqdm import tqdm  # progress bar
from file_functions import load_files, write_lammps_data


def unwrap_coords(first_point, second_point, box_size):
    """Adjust first coordinates to minize distance, accounting for periodic boundary conditions.

    Args:
        first_point (tuple): x, y, z coordinates (floats) of first point
        second_point (tuple): x, y, z coordinates (floats) of first point
        box_size (float): length of simulation box, assumes cubic

    Returns:
        np.array: adjusted x, y, z coordinates of first point
    """
    first_point = np.array(first_point)
    second_point = np.array(second_point)
    diff = abs(first_point - second_point)
    adjusted_first_point = first_point.copy()

    adjusted_first_point[(diff > box_size/2) & (first_point < second_point)] += box_size
    adjusted_first_point[(diff > box_size/2) & (first_point > second_point)] -= box_size

    return adjusted_first_point


def wrap_coords(point, box_size):
    """Adjust coordinates to be within simulation box.

    Args:
        point (np.array): x, y, z, coordinates of point
        box_size (float): length of side of cubic simulation box

    Returns:
        np.array: adjusted x, y, z coodinates of point
    """
    adjusted_point = np.array(point, dtype='float64') % box_size
    return adjusted_point


def create_neighborhood(x, y, z, distance=1, intervals=9):
    """Create neighborhood at intervals of phi and theta

    Args:
        x (np.float): x coordinate of current bead
        y (np.float): y coordinate of current bead
        z (np.float): z cooedinate
        distance (int, optional): Bond length. Defaults to 1.
        intervals (int, optional): Interval for phi and theta. Defaults to 9.

    Returns:
        np.array: x, y, z coordinates for all neighbors
    """
    points = []
    # Adjust intervals for finer granularity
    for theta in np.linspace(0, 2*np.pi, intervals):
        for phi in np.linspace(0, np.pi, intervals):

            xn = x + distance * np.sin(phi) * np.cos(theta)
            yn = y + distance * np.sin(phi) * np.sin(theta)
            zn = z + distance * np.cos(phi)

            points.append([xn, yn, zn])

    return np.array(points)


def check_distances(first_array, second_array, cutoff, box_size):
    """
    Check if any row in first_array is within a given distance of any row in array B,
    considering periodic boundary conditions.

    Args:
        first_array: A NumPy array of shape (N, 3) representing the coordinates of points.
        second_array: A NumPy array of shape (M, 3) representing the coordinates of other points.
        cutoff: The maximum distance for a point to be considered "close".
        box_size: The size of the periodic box in each dimension (assumed to be cubic).

    Returns:
        A boolean array of shape (N,) indicating whether each row in A is close to any row in B.
    """

    # Calculate all pairwise distances
    dx = np.abs(first_array[:, np.newaxis, 0] - second_array[:, 0])
    dy = np.abs(first_array[:, np.newaxis, 1] - second_array[:, 1])
    dz = np.abs(first_array[:, np.newaxis, 2] - second_array[:, 2])

    # Apply periodic boundary conditions
    dx = np.minimum(dx, box_size - dx)
    dy = np.minimum(dy, box_size - dy)
    dz = np.minimum(dz, box_size - dz)

    # Calculate distances
    distances = np.sqrt(dx**2 + dy**2 + dz**2)

    # Check if any distance is less than the threshold
    is_close = np.any(distances < cutoff, axis=1)

    return is_close


def check_neighborhood(neighborhood, bead_positions, box_size, cutoff=0.5):
    """Get unoccupied neighborhood.

    Args:
        neighborhood (np.array): x, y, z coordinates of all neighbors
        bead_positions (np.array): x, y, z coordinates of current bead
        box_size (float): length of cubic simulation box
        cutoff (float, optional): Distance within which beads considered overlapping.
            Defaults to 0.5.

    Returns:
        np.array: neighborhood without occupied spots, False if all full
    """
    occupied_neighbors = check_distances(neighborhood, bead_positions, cutoff, box_size)
    # Selects all unoccupied spots in neighborhood
    open_neighborhood = neighborhood[~occupied_neighbors]
    return open_neighborhood


def find_neighborhood(x, y, z, bead_positions, box_size):
    """ Find target sites for next bead, removing occupied sites.

    Args:
        x (float): x coordinate of current bead
        y (float): y coordinate of current bead
        z (float): z coordinate of current bead
        bead_positions (np.array): x, y, z coordinates of beads already created for current chain
        box_size (float): length of cubic simulation box

    Raises:
        ValueError: Initial and expanded neighborhood are fully occupied

    Returns:
        np.array: x, y, z coordinates of unoccupied neighboring sites
    """
    retry_interval = 4
    neighborhood = create_neighborhood(x, y, z)
    open_neighborhood = check_neighborhood(neighborhood, bead_positions, box_size)
    if open_neighborhood.size == 0:
        # Try different interval for neighborhood creation
        neighborhood = create_neighborhood(x, y, z, intervals=retry_interval)
        open_neighborhood = check_neighborhood(neighborhood, bead_positions, box_size)

        if open_neighborhood.size == 0:
            raise ValueError(f"Neighborhood full at coordinates ({x}, {y}, {z}): {neighborhood}")
    return open_neighborhood


def select_indices(theta, open_neighborhood, current_point):
    """ Select neighbors that match desired direction

    Args:
        theta (np.array): probability threshold for each direction
        open_neighborhood (np.array): x, y, z coordinates of available neighbors
        current_point (np.array): x, y, z coordinates of current bead

    Returns:
        np.array: list of indices 
    """
    indices = np.empty_like(open_neighborhood)
    for i in range(open_neighborhood.shape[1]):
        column = open_neighborhood[:, i]
        if random.random() < theta[i]:
            indices[:, i] = column < current_point[i]
        else:
            indices[:, i] = column >= current_point[i]
    return indices


def calculate_theta(current_point, target_point, n, i):
    """Calculate chance of moving in positive or negative direction."""
    x, y, z = current_point
    xtarg, ytarg, ztarg = target_point
    denom = (0.97 / 2.0) * (n - (i * 2))

    theta_x = 0.5 * (1 - ((xtarg - x) / denom))
    theta_y = 0.5 * (1 - ((ytarg - y) / denom))
    theta_z = 0.5 * (1 - ((ztarg - z) / denom))
    return theta_x, theta_y, theta_z


def step_choice(i, current_point, target_point, bead_positions, box_size, n):
    """Choose next step in constrained walk."""
    x, y, z = current_point

    theta = calculate_theta(current_point, target_point, n, i)

    open_neighborhood= find_neighborhood(x, y, z, bead_positions, box_size)
    # rows = np.any(region != 0, axis=1)
    # region = region[rows]
    index = select_indices(theta, open_neighborhood, current_point)

    row_sums = np.sum(index, axis=1)
    max_sum = np.max(row_sums)
    options = np.where(row_sums == max_sum)[0]
    choice = np.random.choice(options)
    if not np.any(open_neighborhood[choice]):
        print(choice)
    return open_neighborhood[choice]


def constrained_walk(start, end, box_size, n=15):
    """Generate atom position in a constrained walk between two nodes."""
    [x0, y0, z0] = start
    [xn, yn, zn] = end
    [x0, y0, z0] = unwrap_coords(start, end, box_size)
    x = x0
    y = y0
    z = z0
    bead_positions = np.empty((n+2, 3))
    bead_positions[:] = np.nan
    bead_positions[0] = [x0, y0, z0]
    bead_positions[-1] = [xn, yn, zn]

    for i in range(int(n/2)+1):
        # Step from starting side
        current_point = x, y, z
        target_point = xn, yn, zn
        current_point = unwrap_coords(current_point, target_point, box_size)
        x, y, z = step_choice(i, current_point, target_point,
                              bead_positions, box_size, n)
        bead_positions[i+1] = [x, y, z]
        # Step from ending side
        current_point = xn, yn, zn
        target_point = x, y, zn
        xn, yn, zn = step_choice(i, current_point, target_point,
                                 bead_positions, box_size, n)
        bead_positions[n-i] = [xn, yn, zn]
    return bead_positions[1:-1]


def calculate_wrapped_distance(array, box_size):
    """Calculate the shortest distance between two points"""
    num_rows = array.shape[0]
    results = np.zeros_like(array)

    for i in range(0, num_rows - 1, 1):
        new_row1 = unwrap_coords(array[i], array[i + 1], box_size)

        # Calculate the difference between new_row1 and row i+1
        diff = abs(new_row1 - array[i + 1])
        results[i] = diff

    return np.array(results)


def calculate_wrapped_distance_full(array, box_size):
    """Calculate actual distance, not difference."""
    num_rows = array.shape[0]
    results = np.zeros((num_rows, 1))
    for i in range(0, num_rows - 1, 1):
        new_row = unwrap_coords(array[i], array[i+1], box_size)
        results[i] = math.sqrt((new_row[0]-array[i + 1, 0])**2 +
                               (new_row[1] - array[i + 1, 1])**2 +
                               (new_row[2] - array[i + 1, 2])**2)
    return results


def create_atom_list(node_data, edge_data, len_of_chain):
    """Create panda Data Frame of generated positions and nodes."""
    [x_data, y_data, z_data, node_type] = [node_data[:, 0], node_data[:, 1],
                                           node_data[:, 2], node_data[:, 3]]
    atom_list = pd.DataFrame(data={'ID':
                                   np.arange(0, ((len_of_chain) *
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


def update_bead_list(bead_data, bead_ids, path, chain):
    """Update bead list with beads frm current chain."""
    bead_data.loc[bead_ids, "X"] = path[:, 0]
    bead_data.loc[bead_ids, "Y"] = path[:, 1]
    bead_data.loc[bead_ids, "Z"] = path[:, 2]
    bead_data.loc[bead_ids, "Mol"] = chain + 1
    bead_data.loc[bead_ids, "atom-type"] = 2
    return bead_data


def generate_chain_path(point_0, point_n, cutoff, len_of_chain, box_size):
    """Try to generate chain path within a number of cycles 
    where middle beads are within a cutoff distance."""
    maxdist = 50  # arbitrary large value to start
    cycle = 0
    masterpath = np.empty((1, 1))
    while maxdist > cutoff and cycle < 100:  # arbitrary cut offs
        path = constrained_walk(start=point_0, end=point_n,
                                box_size=box_size, n=len_of_chain)
        curr = max(calculate_wrapped_distance_full(path, box_size))[0]
        cycle += 1
        if curr < maxdist:
            maxdist = curr
            masterpath = path
        maxdist = min(maxdist, curr)
    for i, coords in enumerate(masterpath):
        masterpath[i] = wrap_coords(coords, box_size)
    return masterpath, maxdist, cycle

def generate_and_update(**kwargs):
    """Helper function to generate and update chain data."""
    edge = kwargs['edge']
    chain_index = kwargs['chain_index']
    bond_data = kwargs['bond_data']
    bead_data = kwargs['bead_data']
    node_data = kwargs['node_data']
    box_size = kwargs['box_size']
    len_of_chain = kwargs['len_of_chain']

    point_0 = (node_data[int(edge[0]), 0], node_data[int(edge[0]), 1], node_data[int(edge[0]), 2])
    point_n = (node_data[int(edge[1]), 0], node_data[int(edge[1]), 1], node_data[int(edge[1]), 2])

    masterpath, maxdist, cycle = generate_chain_path(point_0, point_n, 1.3, len_of_chain, box_size)

    id_range = np.arange(len(node_data) + (chain_index * len(masterpath)),
                         len(node_data) + ((chain_index + 1) * len(masterpath)))
    bead_data_updated = update_bead_list(bead_data.copy(), id_range, masterpath, chain_index)
    bond_data_updated = update_bond_list(id_range, edge, bond_data)

    return bead_data_updated, bond_data_updated, cycle, maxdist


def generate_wrapper(task):
    """Wrapper for multiprocessing chain creation."""
    return generate_and_update(**task)


def create_chain_parallel(full_edge_data, bond_data, bead_data, node_data, box_size,
                          len_of_chain, num_processes):
    """Create chains in parallel."""
    with multiprocessing.Pool(processes=num_processes) as pool:
        common_kwargs = {  # Common arguments for all calls
            'bond_data': bond_data,
            'bead_data': bead_data,
            'node_data': node_data,
            'box_size': box_size,
            'len_of_chain': len_of_chain
            }
        tasks = [{'edge': edge, 'chain_index': i, **common_kwargs}
                 for i, edge in enumerate(full_edge_data)]
        results = list(tqdm(pool.imap_unordered(generate_wrapper, tasks),  # Use the wrapper
                                total=len(full_edge_data),
                                desc="Generating Chains"))

    _, _, cycles, maxdists = zip(*results)
    run_info_array = np.column_stack((cycles, maxdists))

    # Update bond_data and bead_data based on results
    for bead_data_updated, bond_data_updated, _, _ in results:
        bead_data.update(bead_data_updated)
        bond_data = pd.concat([bond_data, bond_data_updated], ignore_index=True)

    return bead_data, bond_data, run_info_array


if __name__ == '__main__':
    STUDY_NAME = '20241120B1C1'
    COORDS = ['x', 'y', 'z']
    cpu_num = 1 # int(np.floor(multiprocessing.cpu_count()/2))

    [NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = load_files(STUDY_NAME, COORDS)
    FullEdges = np.concatenate((Edges, PB_edges))
    BeadData = create_atom_list(NodeData, FullEdges, LENGTH_OF_CHAIN)
    BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
    BeadData, BondData, runInfo = create_chain_parallel(FullEdges, BondData, BeadData,
                                                NodeData, BOX_SIZE, LENGTH_OF_CHAIN, cpu_num)
    write_lammps_data(STUDY_NAME, BeadData, BondData, BOX_SIZE)
