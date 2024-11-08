# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:10:06 2024

@author: Natalie.Buchanan
"""

import os
import pandas as pd
import numpy as np


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
    len_of_chain = int(np.loadtxt(os.path.join(base_path, name[0:-3] +
                                               '-n.dat'))) - 1
    node_files = [core_x, core_y, node_type]

    return [node_files, edges, pb_edges, box_size, len_of_chain]


def write_LAMMPS_data(name: str, atom_data, bond_data, box_size,
                      network="auelp"):
    """Write LAMMPS in.data file including positions and bonds."""
    base_path = os.path.join(os.getcwd(), "Data", network)
    data_file = os.path.join(base_path, name + '-in.data')
    with open(data_file, "w",  encoding="utf-8") as file:
        file.write("""{name}

{num_atoms} atoms
{num_bonds} bond

0 {L_box} xlo xhi
0 {L_box} ylo yhi

Masses\n\n""".format(name=name, num_atoms=len(atom_data),
                   num_bonds=len(bond_data), L_box=box_size))
        for i in atom_data["atom-type"].drop_duplicates().sort_values():
            file.write(f'{int(i)} 1\n')
        file.write("""
Atoms # ID molID type x y z\n\n""")
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
                       f"{int(bond_data.iloc[i]['Atom1'])} " +
                       f"{int(bond_data.iloc[i]['Atom2'])} " +
                       "\n")


def create_atom_list(node_data, edge_data, length_of_chain):
    """Create panda Data Frame of generated positions and nodes."""
    x_data, y_data, node_type = node_data
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



def create_atom_list_3D(node_data, edge_data, length_of_chain):
    """Create panda Data Frame of generated positions and nodes."""
    [x_data, y_data, z_data, node_type] = [node_data[:,0], node_data[:,1],
                                           node_data[:,2], node_data[:,3]]
    atom_list = pd.DataFrame(data={'ID':
                                   np.arange(0, ((length_of_chain) *
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

