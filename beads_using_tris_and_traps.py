# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:20:35 2024

@author: Natalie.Buchanan
"""
import random
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import file_functions


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
    # subtract 1 to get number of connecting beads from number of bonds
    len_of_chain = int(np.loadtxt(os.path.join(base_path, name[0:-3] +
                                               '-n.dat'))) - 1
    node_files = np.asarray([core_x, core_y, core_z, node_type])
    node_files = node_files.transpose()

    return [node_files, edges, pb_edges, box_size, len_of_chain]


def upside_down_trapezoid(start, end, number_of_beads, short_base):
    possible_shapes = []
    for long_base in np.arange(1, (number_of_beads + short_base)/2, 2):
        side = (number_of_beads - long_base)/2
        if (long_base%1 == 0) and (side%1 == 0):
            possible_shapes.append([long_base, side])
    long_base, side = random.choice(possible_shapes)

    height = 1/2*np.sqrt((4*(side**2)) - ((long_base-short_base)**2))
    alpha = np.arcsin(height/side)

    base_vector = end - start
    base_length = np.linalg.norm(base_vector)
        
    if abs(base_length - short_base) > 1e-6:
      print("Error: provided base length does not match side_lengths[0]")
      return None
    
    normal_vector = np.cross(base_vector, np.array([0, 0, 1]))
    if np.allclose(normal_vector, np.array([0,0,0])):
      normal_vector = np.cross(base_vector, np.array([0, 1, 0]))

    normal_vector = normal_vector / np.linalg.norm(normal_vector) 
    print(normal_vector)

    delta = np.cos(alpha)
    v3 = start + delta*base_vector + height*normal_vector
    v4 = end + delta*base_vector + height*normal_vector
    points = [start, v3, v4, end]
    lengths = [short_base, long_base, side]
    return points, lengths


def create_positions(points, lengths, number_of_beads):
    [start, v3, v4, end] = points
    [short_base, long_base, side] = lengths
    AtomPositions=pd.DataFrame(data=np.zeros([number_of_beads+2, 3]), columns=["X", "Y", "Z"], index = np.arange(0, number_of_beads+2))
    # BUG: Beads are off a position. Nodes should be 0 and N+1. Currently last node is N
    generated_positions = np.linspace(start, v3, int(side)+1)
    for i in range(int(side)+1):
        print(i)
        AtomPositions.loc[i, "X"] = generated_positions[i, 0]
        AtomPositions.loc[i, "Y"] = generated_positions[i, 1]
        AtomPositions.loc[i, "Z"] = generated_positions[i, 2]

    generated_positions = np.linspace(v3, v4, int(long_base) + 2)
    for i in range(int(long_base)+1):
        print(int(side) + i)
        AtomPositions.loc[int(side) + i, "X"] = generated_positions[i, 0]
        AtomPositions.loc[int(side) + i, "Y"] = generated_positions[i, 1]
        AtomPositions.loc[int(side) + i, "Z"] = generated_positions[i, 2]
    
    generated_positions = np.linspace(v4, end, int(side) + 1)
    for i in range(int(side)+1):
        print(int(side) + int(long_base) + i)
        AtomPositions.loc[int(side) + int(long_base) + i, "X"] = generated_positions[i, 0]
        AtomPositions.loc[int(side) + int(long_base) + i, "Y"] = generated_positions[i, 1]
        AtomPositions.loc[int(side) + int(long_base) + i, "Z"] = generated_positions[i, 2]
    print(AtomPositions)

def trapezoid():
    pass

def init_shape_creation(start, end, number_of_beads, box_size):
    """Generate a chain that forms a trapezoid or triangle."""
    vector = end - start
    dx = np.abs(end[0] - start[0])
    dy = np.abs(end[1] - start[1])
    dz = np.abs(end[2] - start[2])

    # Periodic boundary conditions
    dx = np.minimum(dx, box_size - dx)
    dy = np.minimum(dy, box_size - dy)
    dz = np.minimum(dz, box_size - dz)

    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    print(vector, dist)
    bond_length = 0.97

    if number_of_beads % 2:
        print('ODD')
        if (dist > 0) and (dist <= bond_length):
            points = upside_down_trapezoid()
        elif dist > bond_length <= 2*bond_length:
            points = trapezoid()
        
    else:
        print('EVEN')
    return vector

def select_chain(full_edge_data, bond_data, bead_data, node_data):
    """Create chains."""
    [node_x, node_y, node_z, node_type] = [node_data[:,0], node_data[:,1],
                                           node_data[:,2], node_data[:,3]]
    colors = mpl.colormaps['rainbow'](np.linspace(0, 1, len(full_edge_data)))
    plt.figure()
    fullcycle = np.zeros([len(full_edge_data), 2])
    for chain, edge in enumerate(full_edge_data):
        point_0 = np.array([node_x[int(edge[0])], node_y[int(edge[0])],
                   node_z[int(edge[0])]])
        point_n = np.array([node_x[int(edge[1])], node_y[int(edge[1])],
                   node_z[int(edge[1])]])
        maxdist = 50  # arbitrary large value to start
        cycle = 0
        while maxdist > 3 and cycle < 25:  # arbitrary cut offs
            path = init_shape_creation(start=point_0, end=point_n,
                             number_of_beads=LENGTH_OF_CHAIN, box_size=BOX_SIZE)
            path_x = path[:, 0]
            path_y = path[:, 1]
            curr = 1
            # curr = max([max(calculate_wrapped_distance(path_x)),
            #             max(calculate_wrapped_distance(path_y))])
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
        # bond_data = update_bond_list(id_range, edge, bond_data)

        # if maxdist > 3:
        #     plt.gca().set_aspect('equal')
        #     plt.show()
        #     print(max(calculate_wrapped_distance(path_x)),
        #           max(calculate_wrapped_distance(path_y)))
        #     raise ValueError('Error in chain ' + str(chain))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Path")
    plt.gca().set_aspect('equal')
    plt.show()
    return bead_data, bond_data

# %%

STUDY_NAME = '20241016B1C1'

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = file_functions.create_atom_list_3D(NodeData, FullEdges,
                                           LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
step = 1/(2*np.sqrt(3))
points, lengths = upside_down_trapezoid(np.array([1, 1, 5]), np.array([1+step, 1+step, 5+step]),25, 0.5)
create_positions(points, lengths, 25)
# BeadData, BondData = select_chain(FullEdges, BondData, BeadData, NodeData)
