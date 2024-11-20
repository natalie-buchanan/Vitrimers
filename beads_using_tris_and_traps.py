# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:20:35 2024

@author: Natalie.Buchanan
"""
# %%
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import file_functions
from beads_from_edge_3d import create_atom_list

# %%

def adjust_coordinates_periodic(start, end, box_size):
    """ Adjust coordinates to account for periodic boundary conditions.

    If the distance between points is greater when wrapping around the boundary, 
    then shift the 'end' coordinate by the box size to wrap it.

    Args:
        start (np.array): x, y, z coordinates of first point
        end (np.array): x, y, z coordinates of second point
        box_size (np.array): length of simulation box, assuming cube

    Returns:
        np.array: adjusted x, y, z coordinates of second point
    """

    raw_diff = np.abs(end - start)
    wrapped_diff = box_size - raw_diff
    end = np.where(raw_diff > wrapped_diff, end - box_size, end)
    return end


def wrap_coords(first_point, box_size):
    """ If point outside the simulation box, adjusts by box size.

    Args:
        first_point (np.array): x, y, z coordinates of point
        box_size (np.array): length of side of simulation box, assuming cubic in I octant

    Returns:
        np.array: Adjusted x, y, z coordinates
    """
    wrapped_coords = np.array(first_point, dtype='float64')

    # Increase negative coordinates
    while np.any(np.less(wrapped_coords, 0)):
        wrapped_coords[wrapped_coords < 0] += box_size

    # Decrease coordinates greater than box
    while np.any(np.greater(wrapped_coords, box_size)):
        wrapped_coords[wrapped_coords > box_size] -= box_size

    return wrapped_coords


def _find_trapezoid_vertices(first_vertex, fourth_vertex, lengths):
    """Find location of vertices.

    Args:
        first_vertex (np.array): x, y, z coordinates of first node
        fourth_vertex (np.array): x, y, z coordinates of second node
        lengths (list): length of sides in order: short, side, long

    Returns:
        list: list of np.array containing x, y, z coordinates of all vertices in order
    """
    short_base_length, side_length, long_base_length = lengths

    # Find height and angle between side and longer base using geometry
    height = 1/2*np.sqrt((4*(side_length**2)) -
                         ((long_base_length-short_base_length)**2))
    base_side_angle = np.arcsin(height/side_length)

    # Find vector between the nodes
    base_vector = fourth_vertex - first_vertex
    unit_base_vector = base_vector/np.linalg.norm(base_vector)

    # Find vector perpendicular to trapezoid plane using cross product with z-axis
    normal_vector = np.cross(base_vector, np.array([0, 0, 1]))

    # If base vector is close to parallel with z-axis,
    # find normal vector using cross product with y-axis instead
    if np.allclose(normal_vector, np.array([0, 0, 0])):
        normal_vector = np.cross(base_vector, np.array([0, 1, 0]))

    # Normalize vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    # Find location of other vertices
    # horizontal distance from node to vertex
    horizontal_distance = side_length * np.cos(base_side_angle)
    second_vertex = first_vertex + \
        (horizontal_distance*unit_base_vector) + (height*normal_vector)
    third_vertex = fourth_vertex - \
        (horizontal_distance*unit_base_vector) + (height*normal_vector)
    points = [first_vertex, second_vertex, third_vertex, fourth_vertex]
    return points


def upside_down_trapezoid(first_vertex, fourth_vertex, number_of_beads, short_base_length):
    """Find vertices and side lengths to create a trapezoid where the short side is 
    between two nodes.

    Args:
        first_vertex (np.array): x, y, z coordinates of first node
        fourth_vertex (np.array): x, y, z coordinates of second node
        number_of_beads (int): number of beads in connecting chain, not counting nodes
        short_base_length (np.float): distance between two nodes

    Returns:
        list: list of vertices in order [1, 2, 3, 4]
              list of lengths in order [1-4, 2-3, 3-4]
    """
    # Finds possible shapes that meet criteria
    number_of_bonds = number_of_beads + 1
    possible_shapes = []
    for long_base_length in np.arange(1, (number_of_bonds + short_base_length)/2, 2):
        side_length = (number_of_bonds - long_base_length)/2
        # Shape only accepted if all three segements will contain whole number of beads
        if (long_base_length % 1 == 0) and (side_length % 1 == 0):
            possible_shapes.append([long_base_length, side_length])
    # Randomly pick a set of dimensions
    long_base_length, side_length = random.choice(possible_shapes)
    lengths_by_position = [short_base_length, long_base_length, side_length]
    lengths_by_size = [short_base_length, side_length, long_base_length]
    points = _find_trapezoid_vertices(
        first_vertex, fourth_vertex, lengths_by_size)
    return points, lengths_by_position


def trapezoid(first_vertex, fourth_vertex, number_of_beads, long_base_length):
    """Find vertices and side lengths of triangle or trapezoid with long base between nodes

    Args:
        first_vertex (np.array): x, y, z coordinates of first node
        fourth_vertex (np.array): x, y, z coordinates of second node
        number_of_beads (int): number of beads in chain, not including nodes
        long_base (np.float): distance between two nodes

    Raises:
        ValueError: no possible shapes found

    Returns:
        list: list of vertices in order [1, 2, 3, 4]
              list of lengths in order [1-4, 2-3, 3-4]
    """
    # Find possible shapes
    possible_shapes = []
    number_of_bonds = number_of_beads + 1
    if number_of_bonds % 2:  # odd
        short_base_options = np.arange(1, long_base_length, 2)
    else:
        short_base_options = np.arange(0, long_base_length, 2)
    if len(short_base_options) == 0:
        short_base_options = [0]
    for short_base_length in short_base_options:
        side_length = (number_of_bonds - short_base_length)/2
        # Possible only if all sides made up of whole number of beads
        if (short_base_length % 1 == 0) and (side_length % 1 == 0):
            possible_shapes.append([short_base_length, side_length])
    if len(possible_shapes) == 0:
        raise ValueError('Did not find possible shapes.')
    short_base_length, side_length = random.choice(possible_shapes)

    lengths_by_position = [long_base_length, short_base_length, side_length]
    lengths_by_size = [short_base_length, side_length, long_base_length]
    points = _find_trapezoid_vertices(
        first_vertex, fourth_vertex, lengths_by_size)

    return points, lengths_by_position


def create_positions(points, lengths, number_of_beads):
    """Generates evenly spaced 3D coordinates between vertices

    Args:
        points (list): the four vertices of the shape in order
        lengths (list): list of lengths between vertices in order 1-4, 2-3, and 3-4
        number_of_beads (int): the total number of beads in the chain, not counting nodes

    Returns:
        pandas.DataFrame: Dataframe with X, Y, Z positions of current chain
    """

    [first_vertex, second_vertex, third_vertex, fourth_vertex] = points
    # Base_2_3 is length between second and third vertices
    # Side_length is the the length of side 1-2 and 3-4, assumed to be equal
    base_2_3, side_length = lengths[1], lengths[2]
    bead_positions = pd.DataFrame(data=np.zeros(
        [number_of_beads+2, 3]), columns=["X", "Y", "Z"], index=np.arange(0, number_of_beads+2))

    # Positions between vertex 1 and 2, including the vertices
    generated_positions = np.linspace(
        first_vertex, second_vertex, int(side_length)+1)
    for i in range(int(side_length)+1):
        bead_positions.loc[i, "X"] = generated_positions[i, 0]
        bead_positions.loc[i, "Y"] = generated_positions[i, 1]
        bead_positions.loc[i, "Z"] = generated_positions[i, 2]

    # Positions between vertex 2 and 3, overwriting 2 and creating 3
    generated_positions = np.linspace(
        second_vertex, third_vertex, int(base_2_3) + 2)
    for i in range(int(base_2_3)+1):
        bead_positions.loc[int(side_length) + i,
                           "X"] = generated_positions[i, 0]
        bead_positions.loc[int(side_length) + i,
                           "Y"] = generated_positions[i, 1]
        bead_positions.loc[int(side_length) + i,
                           "Z"] = generated_positions[i, 2]

    # Positions between vertex 3 and 4, overwriting 3 and creating 4
    generated_positions = np.linspace(
        third_vertex, fourth_vertex, int(side_length) + 1)
    for i in range(int(side_length) + 1):
        bead_positions.loc[int(side_length) + int(base_2_3) + i,
                           "X"] = generated_positions[i, 0]
        bead_positions.loc[int(side_length) + int(base_2_3) + i,
                           "Y"] = generated_positions[i, 1]
        bead_positions.loc[int(side_length) + int(base_2_3) + i,
                           "Z"] = generated_positions[i, 2]

    return bead_positions[1:-1]


def init_shape_creation(start, end, number_of_beads, box_size):
    """Initialize node positions and calls shape based on separation.

    Args:
        start (np.array): x, y, z coordinates of first node
        end (np.array): x, y, z coordinates of second node
        number_of_beads (int): length of connecting chains, excluding nodes
        box_size (np.array): length of simulation box side, assuming cubic

    Raises:
        ValueError: Number of beads created is different then defined

    Returns:
        pd.DataFrame: X, Y, Z positions of beads in current chain
    """

    # Ensures nodes are within simulation box
    start = wrap_coords(start, box_size)
    end = wrap_coords(end, box_size)

    # Adjusts coordinates of second node by box size to minimize distance
    raw_diff = np.abs(end - start)
    wrap_diff = np.abs(BOX_SIZE - raw_diff)
    dx, dy, dz = np.minimum(raw_diff, wrap_diff)
    end = np.where(raw_diff < wrap_diff, end, start + wrap_diff)

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    bond_length = 1

    # If even number of beads with close separation of nodes, generate
    # upside down trapezoid, with short base connecting the nodes
    if not number_of_beads % 2 and (dist > 0) and (dist <= bond_length):
        points, lengths = upside_down_trapezoid(
            start, end, number_of_beads, dist)
    else:
        # Generate trapezoid with the longer base between the nodes
        points, lengths = trapezoid(start, end, number_of_beads, dist)
    bead_positions = create_positions(points, lengths, number_of_beads)

    if len(bead_positions) != number_of_beads:
        raise ValueError(
            'Expected {number_of_beads}. Created {len(bead_positions)}')
    return bead_positions


def update_bond_list(bead_ids, node_id, bond_list):
    """ Add bonds in current chain to full list of bonds

    Args:
        bead_ids (np.array): index of beads
        node_id (np.array): node ID numbers for current chain
        bond_list (pd.DataFrame): information to write lammps in.data file bonds section

    Returns:
        pd.DataFrame: information to write lammps in.data file bonds section
    """
    # Add ID of first/second node to start/end of list of atom IDs for beads in current chain
    bead_ids = np.concatenate(([node_id[0]], bead_ids, [node_id[1]]))
    current_list = pd.DataFrame({"BondType": 1, "Atom1": bead_ids[0:-1],
                                 "Atom2": bead_ids[1:]})
    bond_list = pd.concat([bond_list, current_list], ignore_index=True)
    return bond_list


def update_bead_list(bead_data, bead_ids, path, chain):
    """ Update dataframe for LAMMPS in.data atoms section with info of current chain.

    Args:
        bead_data (pd.DataFrame): atom ID, X, Y, Z, Molecule ID, and atom type
        bead_ids (np.array): IDS of atoms in current chain
        path (pd.DataFrame): X, Y, Z coordinates of current chain
        chain (int): ID of chain being built, starting at 0

    Returns:
        pd.DataFrame: Updated dataframe including current chain information
    """
    # Adjust ids to match start of the chain
    shifted_ids = path.index.values + (bead_ids - path.index.values)

    path["ID"] = shifted_ids
    path.set_index("ID", inplace=True)

    bead_data.update(path)
    bead_data.loc[bead_ids, "Mol"] = chain + 1

    # Assume all connecting beads are Type 2
    bead_data.loc[bead_ids, "atom-type"] = 2
    return bead_data


def create_chains(full_edge_data, bond_data, bead_data, node_data, box_size, len_of_chain):
    """ Create atom positions from graphical input data for use in LAMMPS in.data file.

    Args:
        full_edge_data (np.array): Each row defines chain with IDS of starting and ending node
        bond_data (pd.DataFrame): List of bonds: bond type, starting bead ID, ending bead ID
        bead_data (pd.DataFrame): List of beads: ID, X, Y, Z, Molecule ID, Atom Type
        node_data (np.array): Coordinates and types of node in index order
        box_size (np.array): Length of side of simulation box, assuming cubic shape
        len_of_chain (int): Number of beads in each connecting chain, excluding nodes

    Returns:
        list: Updated bead_data and bond_data for current system
    """

    # Sets up 3D plot
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    chain = -1
    for chain, edge in enumerate(full_edge_data):
        print(chain)  # Tracks progress
        # Define starting and ending nodes
        point_0 = np.asarray([node_data[int(edge[0]), 0], node_data[int(edge[0]), 1],
                              node_data[int(edge[0]), 2]])
        point_n = np.asarray([node_data[int(edge[1]), 0], node_data[int(edge[1]), 1],
                              node_data[int(edge[1]), 2]])

        # Generate chain path and adjust for periodic boundaru conditions
        masterpath = init_shape_creation(
            point_0, point_n, len_of_chain, box_size)
        masterpath = masterpath.apply(wrap_coords, axis=1, result_type='broadcast',
                                      box_size=box_size)

        # Generate bead IDs
        starting_index = len(node_data) + (chain * len(masterpath))
        id_range = np.arange(starting_index, starting_index + len(masterpath))

        # Fill in dataframes with information for current chain
        bead_data = update_bead_list(bead_data, id_range, masterpath, chain)
        bond_data = update_bond_list(id_range, edge, bond_data)

        # Add current chain to plot
        ax.scatter(masterpath["X"], masterpath["Y"], masterpath["Z"],
                   color=colors[chain], marker='o')
        ax.scatter(point_0[0], point_0[1], point_0[2], color='k', marker='>')
        ax.scatter(point_n[0], point_n[1], point_n[2], color='k', marker='>')

    # Show complete plot
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data


# %%
STUDY_NAME = '20241016B1C1'
COORDS = ['x', 'y', 'z']

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = file_functions.load_files(STUDY_NAME,
                                                                                   COORDS)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = create_atom_list(NodeData, FullEdges, LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData = create_chains(FullEdges, BondData, BeadData, NodeData, BOX_SIZE,
                                   LENGTH_OF_CHAIN)
file_functions.write_lammps_data(STUDY_NAME, BeadData, BondData, BOX_SIZE)
