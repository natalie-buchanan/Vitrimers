"""
Created on Mon Oct 28 13:37:25 2024

@author: Natalie.Buchanan
"""

import random
import multiprocessing
import traceback
import pandas as pd
import numpy as np
from tqdm import tqdm  # progress bar
from file_functions import load_files, write_lammps_data, write_lammps_input


def unwrap_coords(first_point, second_point, box_size):
    """Adjust first coordinates to minimize distance, accounting for periodic boundary conditions.

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

    adjusted_first_point[(diff > box_size/2) &
                         (first_point < second_point)] += box_size
    adjusted_first_point[(diff > box_size/2) &
                         (first_point > second_point)] -= box_size

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
    occupied_neighbors = check_distances(
        neighborhood, bead_positions, cutoff, box_size)
    # Selects all unoccupied spots in neighborhood
    open_neighborhood = neighborhood[~occupied_neighbors]
    return open_neighborhood


def find_neighborhood(current_coordinate, bead_positions, box_size):
    """ Find target sites for next bead, removing occupied sites.

    Args:
        current_coordinate (tuple): x, y, z coordinates of current bead
        bead_positions (np.array): x, y, z coordinates of beads already created for current chain
        box_size (float): length of cubic simulation box

    Raises:
        ValueError: Initial and expanded neighborhood are fully occupied

    Returns:
        np.array: x, y, z coordinates of unoccupied neighboring sites
    """
    x, y, z = current_coordinate
    retry_interval = 4
    neighborhood = create_neighborhood(x, y, z)
    open_neighborhood = check_neighborhood(
        neighborhood, bead_positions, box_size)
    if open_neighborhood.size == 0:
        # Try different interval for neighborhood creation
        neighborhood = create_neighborhood(x, y, z, intervals=retry_interval)
        open_neighborhood = check_neighborhood(
            neighborhood, bead_positions, box_size)

        if open_neighborhood.size == 0:
            raise ValueError(f"Neighborhood full at coordinates ({
                             x}, {y}, {z}): {neighborhood}")
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


def calculate_theta(current_point, target_point, number_of_beads, i):
    """Calcaulate change of moving in positive or negative direction.

    Args:
        current_point (np.array): x, y, z coordinates of current bead
        target_point (np.array): x, y, z, coordinate of last bead built on other side of chain
        number_of_beads (int): total number of beads in the chain, excluding nodes
        i (int): Number of beads current side is away from node

    Returns:
        tuple: theta_x, theta_y, theta_z
    """
    x, y, z = current_point
    xtarg, ytarg, ztarg = target_point
    bond_length = 1
    denom = (bond_length / 2.0) * (number_of_beads - (i * 2))
    if number_of_beads-(i*2) == 0:
        print(number_of_beads, i)
        raise ValueError('Denom is 1')

    theta_x = 0.5 * (1 - ((xtarg - x) / denom))
    theta_y = 0.5 * (1 - ((ytarg - y) / denom))
    theta_z = 0.5 * (1 - ((ztarg - z) / denom))
    return theta_x, theta_y, theta_z


def step_choice(i, current_point, target_point, bead_positions, sim_params):
    """Choose where the next bead should be placed.

    Args:
        i (int): distance (in beads) from current bead to node
        current_point (np.ndarray): x, y, z coordinates of bead represented
            as a numpy array of floats
        target_point (tuple): x, y, z coordinates of bead on other side of chain
            represented as a numpt array of floats
        bead_positions (np.ndarray): x, y, z coordinates for all beads in the chain
        sim_params (list): contains box_size (float) and number of beads (int)

    Returns:
        tuple: x, y, z coordinates (floats) for next bead in chain
    """

    box_size, number_of_beads = sim_params

    # Find probability of direction of movement for a constrained walk
    theta = calculate_theta(current_point, target_point, number_of_beads, i)

    # Generate array of neighbors that are not currently occupied
    open_neighborhood = find_neighborhood(
        current_point, bead_positions, box_size)

    # Generate boolean array of whether neighbor is in the desired direction
    neighbor_direction_match = select_indices(
        theta, open_neighborhood, current_point)

    # Choose from neighbors that most match desired theta outcome
    # Select neighbors with highest score then choose one at random
    neighbor_scores = np.sum(neighbor_direction_match, axis=1)
    max_score = np.max(neighbor_scores)
    best_neighbors = np.where(neighbor_scores == max_score)[0]
    choice = np.random.choice(best_neighbors)
    return open_neighborhood[choice]


def constrained_walk(start, end, box_size, n):
    """Generate atom positions in a constrained walk between two nodes.

    Args:
        start (tuple): x, y, z coordinates (floats) of first node
        end (tuple): x, y, z coordinates (floats) of second node
        box_size (float): length of side of cubic simulation box
        n (int): Number of beads in connecting chain (excluding nodes)

    Returns:
        np.ndarry: x, y, z coordinates of all beads in chain (excluding nodes)
    """
    # Reposition first node outside box if necessary to minimize distance between nodes
    start = unwrap_coords(start, end, box_size)
    bead_positions = np.empty((n+2, 3))
    bead_positions[:] = np.nan
    bead_positions[0] = start
    bead_positions[-1] = end
    target_point = start

    for i in range(int(n/2)+1):
        # Step from starting side
        current_point = target_point
        target_point = bead_positions[n+1-i]
        # Reposition current bead outside box if necessary to minimize distance between beads
        current_point = unwrap_coords(current_point, target_point, box_size)
        # Pick position for next bead using constrained convergent walk
        if (n-(i*2)) != 0:
            bead_positions[i+1] = step_choice(i, current_point, target_point,
                                              bead_positions, [box_size, n])
        else:
            pass

        # Step from ending side
        if i+1 < n-i:
            current_point = target_point
            target_point = bead_positions[i+1]

            # Pick position for next bead using constrained convergent walk
            bead_positions[n-i] = step_choice(i, current_point, target_point,
                                            bead_positions, [box_size, n])

    return bead_positions[1:-1]


def calculate_wrapped_distance(array, box_size):
    """Calculate distance between coordinates for adjacent beads, adjusting for periodic boundaries

    Args:
        array (np.ndarray): x, y, z coordinates of all points
        box_size (float): length of side of cubic simulation box

    Returns:
        np.array: array of distance between adjusted points
    """

    # Calculate adjusted points for all points except the last one
    adjusted_points = unwrap_coords(array[:-1], array[1:], box_size)

    # Calculate the difference between adjusted points and the next points
    diff = np.abs(adjusted_points - array[1:])

    # Pad the results with a 0 at the beginning to match the original shape
    results = np.pad(diff, ((0, 1), (0, 0)), 'constant', constant_values=0)

    return results


def calculate_wrapped_distance_full(points, box_size):
    """Calculate distance between adjacent beads, adjusting for periodic boundaries

    Args:
        points (np.ndarray): x, y, z coordinates (floats) of beads in chain
        box_size (float): length of one size of cubic simulation box

    Returns:
        np.ndarray: distance between adjacent points
    """

    # Shift array to get coordinates of the next point
    next_points = np.roll(points, -1, axis=0)

    # Apply unwrapping logic to next_points
    unwrapped_points = unwrap_coords(points, next_points, box_size)

    # Calculate distance using broadcasting
    distances = np.sqrt(np.nansum((unwrapped_points - next_points)**2, axis=1))

    return distances[:-1].reshape(-1, 1)


def create_atom_list(node_data, len_of_chain):
    """Create pd.DataFrame to store bead information and preload with nodes.

    Args:
        node_data (np.ndarray): x, y, z and types of nodes
        edge_data (np.ndarray): array with start and end node IDs for each connecting chain
        len_of_chain (int): number of beads in connecting chain excluding nodes

    Returns:
        pd.DataFrame: DataFrame with x, y, z, molecule id, and atom-type for all beads
            Information from the nodes is filled in, rest of beads in np.nan
    """
    num_nodes = node_data.shape[0]
    total_atoms = np.sum(len_of_chain) + num_nodes

    # Create empty DataFrame
    atom_list = pd.DataFrame({
        'ID': np.arange(total_atoms),
        'X': np.nan,
        'Y': np.nan,
        'Z': np.nan,
        'Mol': np.nan,
        'atom-type': np.nan
    })

    # Add node information to dataframe
    atom_list.loc[:num_nodes - 1, ['X', 'Y', 'Z']] = node_data[:, :3]
    atom_list.loc[:num_nodes - 1, 'atom-type'] = node_data[:, 3]
    atom_list.loc[:num_nodes - 1, 'Mol'] = 0
    return atom_list


def update_bond_list(bead_ids, node_id, bond_list):
    """Append bonds of current chain to master bond list

    Args:
        bead_ids (np.ndarray): index or ids of beads in current chain
        node_id (np.ndarray): index or ids of both nodes at ends of current chain
        bond_list (pd.DataFrame): bond types, atom 1, and atom 2 to define all bonds

    Returns:
        pd.DataFrame: Updated dataframe with bonds from current chain
    """
    bead_ids = np.insert(bead_ids, [0], node_id[0])
    bead_ids = np.append(bead_ids, node_id[1])
    current_list = pd.DataFrame({"BondType": 1, "Atom1": bead_ids[0:-1],
                                 "Atom2": bead_ids[1:]})
    bond_list = pd.concat([bond_list, current_list], ignore_index=True)
    return bond_list


def update_bead_list(bead_data, bead_ids, path, chain):
    """Update position data frame with location of beads for current chain.

    Args:
        bead_data (pd.DataFrame): ID, x, y, z, molecule id, atom-type for all beads in system
        bead_ids (np.ndarray): atom IDS for beads in current chain
        path (np.ndarray): x, y, z coordinates for all beads in current chain
        chain (int): number of chain generated

    Returns:
        pd.DataFrame: updated position DataFrame
    """
    bead_data.loc[bead_ids, ["X", "Y", "Z"]] = path
    bead_data.loc[bead_ids, "Mol"] = chain + 1
    bead_data.loc[bead_ids, "atom-type"] = 2
    return bead_data


def generate_chain_path(point_0, point_n, cutoff, len_of_chain, box_size):
    """Try to generate path for chain such that middle beads are not further apart then the cutoff.

    Args:
        point_0 (np.ndarray): x, y, z coordinates of first node
        point_n (np.ndarray): x, y, z coordinate of second node
        cutoff (dictionary): Minimum and maximum allowed distance for middle bond
        len_of_chain (int): number of beads in chain, excluding nodes
        box_size (float): length of side of cubic simulation box    

    Raises:
        ValueError: No path is found in the number of cycles alloted 
            that meets distance requirement

    Returns:
        list: bead coordinates for path with lowest separation (np.ndarray)
            the separation between middle beads for this path (float)
            the number of cycles it took to find path that met criteria (int)
    """
    outside_cutoff = True
    cycle = 0  # Initialize loop
    cycle_limit = 10000
    masterpath = np.empty((1, 1))

    # Generate paths until one is found that meets separation criteria
    # or maximum number of cycles is reached
    while outside_cutoff and cycle < cycle_limit:
        # Generate path
        path = constrained_walk(start=point_0, end=point_n,
                                box_size=box_size, n=len_of_chain)
        # Find maximum separation between adjacent beads in current chain
        curr_max = max(calculate_wrapped_distance_full(path, box_size))[0]
        curr_min = min(calculate_wrapped_distance_full(path, box_size))[0]
        cycle += 1
        # If current max separation is smaller than current miminum found
        #  save path and update maxdist
        if curr_max< cutoff["max"] and curr_min > cutoff["min"]:
            outside_cutoff = False
            masterpath = path

    # Create even;y spaced points if no paths meeting found within cycle limit
    if outside_cutoff is True:
        print("Solution not found within cycle limit")
        print(len_of_chain, point_0, point_n)
        masterpath = np.linspace(point_0, point_n, len_of_chain+2)[1:-1]
    # Move generated positions inside the simulation box
    curr_max = max(calculate_wrapped_distance_full(masterpath, box_size))[0]
    curr_min =  min(calculate_wrapped_distance_full(masterpath, box_size))[0]
    for i, coords in enumerate(masterpath):
        masterpath[i] = wrap_coords(coords, box_size)


    return masterpath, [curr_min, curr_max], cycle


def generate_and_update(lock, shared_data, **kwargs):
    """Generate beads in chain and update dataframes.

    Returns:
        tuple: updated bead_data(pd.DataFrame), updated bond_data (pd.DataFrame),
            cycles (int), and maxdist (float)
    """
    edge = kwargs['edge']
    chain_index = kwargs['chain_index']
    node_data = kwargs['node_data']
    len_of_chain = int(kwargs['len_of_chain'][chain_index])
    distance_mismatch = {"min": 0.9, "max": 1.1}

    # Get x, y, z coordinates for current nodes
    point_0 = node_data[int(edge[0])][:-1]
    point_n = node_data[int(edge[1])][:-1]
    wrapped_distance =  calculate_wrapped_distance_full([point_0, point_n], kwargs['box_size'])

    if wrapped_distance> len_of_chain + 1:
        print('Line 484: Wrapped distance longer than chain')
        print('\n',chain_index, len_of_chain+1,  wrapped_distance, '\n')
        if wrapped_distance >= kwargs['box_size']/2:
            print('Wrapping Error?')

    # Find x, y, z coordinates of beads in chain
    masterpath, maxdist, cycle = generate_chain_path(
        point_0, point_n, distance_mismatch, len_of_chain, kwargs['box_size'])

    # Update data
    id_range = np.arange(np.sum(kwargs['len_of_chain'][0:chain_index]) +len(node_data),
                         np.sum(kwargs['len_of_chain'][0:chain_index+1]) +len(node_data),)
    
    with lock:
        bead_data_updated = update_bead_list(
            shared_data['bead_data'], id_range, masterpath, chain_index)
        bond_data_updated = update_bond_list(
            id_range, edge, shared_data['bond_data'])

        shared_data['bead_data'] = bead_data_updated
        shared_data['bond_data'] = bond_data_updated

    return bead_data_updated, bond_data_updated, cycle, maxdist


def generate_wrapper(task):
    """Wrapper for multiprocessing chain creation."""
    try:
        return generate_and_update(**task)
    except ValueError as e:
        print(f"Error in worker process: {e}")
        traceback.print_exc()
        raise e


def create_chain_parallel(full_edge_data, bead_data, bond_data, sim_params, num_processes):
    """Create bead positions for each defined edge in parallel.

    Args:
        full_edge_data (np.ndarray): IDs of the starting and ending nodes for each edge
        bead_data (pd.DataFrame): x, y, z coordinates, molecule id, and atom-type for each bead
            Nodes are filled in, rest is np.nan
        bond_data (pd.DataFrame): initialized dataframe containing
            bondtype, first atom, second atom for each bonded pair
        sim_params (dict): contains node_data (np.ndarray), box_size(float), and len_of_chain (int)
        num_processes (int): number of processors to run across for multiprocessing

    Returns:
        list: completed bead_data (pd.DataFrame), complete bond_data (pd.DataFrame),
            run info (array) with number of cycles and the max distance between 
            two adjacent beads for each chain
    """
    with multiprocessing.Manager() as manager:
        shared_data = manager.dict(
            {'bead_data': bead_data, 'bond_data': bond_data})
        system_params = {'shared_data': shared_data,
                         'node_data': sim_params['node_data'],
                         'box_size': sim_params['box_size'],
                         'len_of_chain': sim_params['len_of_chain']}
        lock = manager.Lock()

        tasks = [{'edge': edge, 'chain_index': i, 'lock': lock, 'shared_data': shared_data, **system_params}
                 for i, edge in enumerate(full_edge_data)]

        with multiprocessing.Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(generate_wrapper, tasks),  # Use the wrapper
                            total=len(full_edge_data),
                            unit="Chain",
                            desc="Generating Chains"))
        bead_data = shared_data['bead_data']
        bond_data = shared_data['bond_data']

    run_info_array = np.empty([])
    results = pd.DataFrame(data=results, columns=["bead_data", "bond_data", "cycles", "dist"])
    # Update bond_data and bead_data based on results
    for index, row in results.iterrows():
        bead_data.update(row["bead_data"])
        bond_data.update(row["bond_data"])
        run_info_array = np.append(run_info_array, row[["cycles", "dist"]])

    return bead_data, bond_data, run_info_array


if __name__ == '__main__':
    STUDY_NAME = '20241219B0C0'
    NETWORK = 'auelp'
    cpu_num =  1 # int(np.floor(multiprocessing.cpu_count()/2))

    [NodeData, FullEdges, BOX_SIZE, LENGTH_OF_CHAIN] = load_files(STUDY_NAME, NETWORK)
    if NodeData.shape[1] == 3:
        NodeData = np.insert(NodeData, 2, np.nan, axis=1)
    BeadData = create_atom_list(NodeData, LENGTH_OF_CHAIN)
    BondData = pd.DataFrame(
        columns=["BondType", "Atom1", "Atom2"], dtype="int")
    simulation_params = {
        'node_data': NodeData,
        'box_size': BOX_SIZE,
        'len_of_chain': LENGTH_OF_CHAIN
    }
    BeadData, BondData, runInfo = create_chain_parallel(FullEdges, BeadData, BondData,
                                                        simulation_params, cpu_num)
    BeadData.fillna(0, axis=1, inplace=True)
    check = BeadData["ID"].isin(BondData["Atom1"]) | BeadData["ID"].isin(BondData["Atom2"])
    if check.all():
        print(check.value_counts())
    write_lammps_data(STUDY_NAME, BeadData, BondData, BOX_SIZE, NETWORK)
    print(f'data.{STUDY_NAME} created')
    write_lammps_input(STUDY_NAME, NETWORK)
