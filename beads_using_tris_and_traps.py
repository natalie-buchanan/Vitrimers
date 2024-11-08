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


def trapezoid(start, end, number_of_beads, dist):
    longer_side = dist
    short_side = int(number)
    leg = (number_of_beads - short_side)/2
    


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
        if dist > 0 & dist <= bond_length:
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
BeadData, BondData = select_chain(FullEdges, BondData, BeadData, NodeData)
