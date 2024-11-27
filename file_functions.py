# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 12:10:06 2024

@author: Natalie.Buchanan
"""

import os
import numpy as np


def load_files(name: str, network="auelp"):
    """
    Retrieve data from aelp-network-topology-synthesis.

    Args:
        name (str): The name of the simulation to be analyzed.
        coords (list): list of strings of coordinate files
        network (str): Type of network generated. Used for directory.

    Returns:
        list: [node_files: array, edges: array, pb_edges: array, box_size: float, len_of_chain: int]
    """
    base_path = os.path.join(os.getcwd(), "Data", network)

    try:
        core_coords = np.loadtxt(os.path.join(base_path, f'{name}.coords'))
        edges = np.loadtxt(os.path.join(
            base_path, name + '-conn_core_edges.dat'))
        pb_edges = np.loadtxt(os.path.join(
            base_path, name + '-conn_pb_edges.dat'))
        node_type = np.loadtxt(os.path.join(base_path, name +
                                            '-node_type.dat'))
        box_size = np.loadtxt(os.path.join(base_path, name[0:-2] + '-L.dat'))
        len_of_chain = int(np.loadtxt(os.path.join(base_path, name[0:-3] +
                                                   '-nu.dat')))
    except (FileNotFoundError, OSError) as e:
        print(f"Error loading data files: {e}")
        return None  # Or handle the error as appropriate
    node_files = np.column_stack((core_coords, node_type.astype(int)))
    len_of_chain -= 1  # number of beads one less than number of edges

    return [node_files, edges, pb_edges, box_size, len_of_chain]


def write_lammps_data(name: str, atom_data, bond_data, box_size,
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
0 {L_box} zlo zhi

Masses\n\n""".format(name=name, num_atoms=len(atom_data),
                   num_bonds=len(bond_data), L_box=box_size))
        for i in atom_data["atom-type"].drop_duplicates().sort_values():
            file.write(f'{int(i)} 1\n')
        file.write("""
Atoms # molecular\n\n""")
        for i in atom_data.index:
            file.write(f"{int(atom_data.iloc[i]['ID']+1)} " +
                       f"{int(atom_data.iloc[i]['Mol']+1)} " +
                       f"{int(atom_data.iloc[i]['atom-type'])} " +
                       f"{atom_data.iloc[i]['X']} " +
                       f"{atom_data.iloc[i]['Y']} " +
                       f"{atom_data.iloc[i]['Z']} " + "\n")
        file.write("""
Bonds\n\n""")
        for i in bond_data.index:
            file.write(f"{i + 1} " +
                       f"{int(bond_data.iloc[i]['BondType'])} " +
                       f"{int(bond_data.iloc[i]['Atom1']) + 1} " +
                       f"{int(bond_data.iloc[i]['Atom2']) + 1} " +
                       "\n")

def write_lammps_input(name: str, network="auelp"):
    """Write LAMMPS in.lammps file including positions and bonds."""
    base_path = os.path.join(os.getcwd(), "Data", network)
    data_file = os.path.join(base_path, name + '-in.lammps')
    with open(data_file, "w",  encoding="utf-8") as file:
        file.write("""# {name}
units   lj
boundary p p p
atom_style molecular
                   
read_data {name}-in.data
                   
velocity all create 1 837
                   
pair_style lj/cut 2.5
pair_coeff * * 1.0 1.0 2.5
                   
bond_style fene
bond_coeff 1 5 1 1 1
special_bonds fene
                   
timestep 0.01
                   
fix ensem all npt temp 1 1 10 iso 0 0 10
                   
thermo_stype custom step temp press density pe ke etotal
thermo 1000
        
                   """.format(name=name))
