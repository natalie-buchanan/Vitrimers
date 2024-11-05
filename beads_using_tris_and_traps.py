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


def trapezoid(start, end, number_of_beads):
    """Generate a chain that forms a trapezoid or triangle."""
    vector = list(end) - list(start)
    test = input("Vector: " + str(vector))


def select_chain(full_edge_data, bond_data, bead_data, node_data):
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
            print('Trap')
            path = trapezoid(start=point_0, end=point_n,
                             number_of_beads=LENGTH_OF_CHAIN)
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


STUDY_NAME = '20241016A1C1'

[NodeData, Edges, PB_edges, BOX_SIZE, LENGTH_OF_CHAIN] = \
    file_functions.load_files(STUDY_NAME)
FullEdges = np.concatenate((Edges, PB_edges))
BeadData = file_functions.create_atom_list(NodeData, FullEdges,
                                           LENGTH_OF_CHAIN)
BondData = pd.DataFrame(columns=["BondType", "Atom1", "Atom2"], dtype="int")
BeadData, BondData = select_chain(FullEdges, BondData, BeadData, NodeData)
