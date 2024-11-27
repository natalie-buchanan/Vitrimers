import numpy as np
from file_io import (
    L_filename_str,
    config_filename_str
)
from simulation_box_utils import L_arg_eta_func
from scipy.spatial import Delaunay
from network_topology_initialization_utils import (
    core_node_tessellation,
    unique_sorted_edges
)
from node_placement import (
    additional_node_seeding,
    hilbert_node_label_assignment
)

def delaunay_filename_str(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> str:
    """Filename prefix associated with Delaunay-triangulated network
    data files.

    This function returns the filename prefix associated with
    Delaunay-triangulated network data files.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    Returns:
        str: The filename prefix associated with Delaunay-triangulated
        network data files.
    
    """
    # This filename prefix convention is only applicable for data files
    # associated with Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "delaunay":
        import sys
        
        error_str = (
            "This filename prefix convention is only applicable for "
            + "data files associated with Delaunay-triangulated "
            + "networks. This filename prefix will only be supplied if "
            + "network = ``delaunay''."
        )
        sys.exit(error_str)
    return config_filename_str(network, date, batch, sample, config)

def delaunay_L(
        network: str,
        date: str,
        batch: str,
        sample: int,
        dim: int,
        b: float,
        n: int,
        eta_n: float) -> None:
    """Simulation box size for Delaunay-triangulated networks.

    This function calculates and saves the simulation box size for
    Delaunay-triangulated networks.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended number of core nodes.
        eta_n (float): Node packing density.
    
    """
    # This calculation for L is only applicable for
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        import sys
        
        error_str = (
            "This calculation for L is only applicable for "
            + "Delaunay-triangulated networks. This calculation will "
            + "only proceed if network = ``delaunay''."
        )
        sys.exit(error_str)
    
    # Calculate and save L
    np.savetxt(
        L_filename_str(network, date, batch, sample),
        [L_arg_eta_func(dim, b, n, eta_n)])

def delaunay_network_topology_initialization(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Network topology initialization procedure for
    Delaunay-triangulated networks.

    This function loads the simulation box size and the core node
    coordinates. Then, this function ``tessellates'' the core nodes
    about themselves, applies Delaunay triangulation to the resulting
    tessellated network via the scipy.spatial.Delaunay() function,
    acquires back the periodic network topology of the core nodes, and
    ascertains fundamental graph constituents (node and edge
    information) from this topology.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" or "swidt" are applicable (corresponding to Delaunay-triangulated networks ("delaunay") and spider web-inspired Delaunay-triangulated networks ("swidt")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Load L
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate configuration filename prefix. This establishes the
    # configuration filename prefix as the filename prefix associated
    # with Delaunay-triangulated network data files, which is reflected
    # in the delaunay_filename_str() function.
    config_filename_prefix = config_filename_str(
        network, date, batch, sample, config)

    # Generate filenames
    coords_filename = config_filename_prefix + ".coords"
    conn_core_edges_filename = (
        config_filename_prefix + "-conn_core_edges" + ".dat"
    )
    conn_pb_edges_filename = config_filename_prefix + "-conn_pb_edges" + ".dat"

    # Call appropriate helper function to initialize network topology
    if (scheme == "random") or (scheme == "prhd") or (scheme == "pdhu"):
        # Load core node coordinates
        coords = np.loadtxt(coords_filename)
    elif scheme == "lammps":
        skiprows_num = 15
        # Load core node coordinates
        coords = np.loadtxt(coords_filename, skiprows=skiprows_num, max_rows=n)
    
    # Actual number of core nodes
    n = np.shape(coords)[0]

    # Core nodes
    core_nodes = np.arange(n, dtype=int)

    # Tessellate the core node coordinates and construct the
    # pb2core_nodes np.ndarray
    tsslltd_coords, pb2core_nodes = core_node_tessellation(
        dim, core_nodes, coords, L)
    
    del core_nodes

    # Apply Delaunay triangulation
    tsslltd_core_delaunay = Delaunay(tsslltd_coords)

    del tsslltd_coords

    # Extract the simplices from the Delaunay triangulation
    simplices = tsslltd_core_delaunay.simplices

    # List for edges of the core and periodic boundary nodes
    tsslltd_core_pb_edges = []

    for simplex in simplices:
        # In two dimensions, each simplex is a triangle
        if dim == 2:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add that edge
            # to the edge list. Duplicate entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            else: pass
        # In three dimensions, each simplex is a tetrahedron
        elif dim == 3:
            node_0 = int(simplex[0])
            node_1 = int(simplex[1])
            node_2 = int(simplex[2])
            node_3 = int(simplex[3])

            # If any of the nodes involved in any simplex edge
            # correspond to the original core nodes, then add those
            # nodes and that edge to the appropriate lists. Duplicate
            # entries will arise.
            if (node_0 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_0, node_1))
            if (node_1 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_1, node_2))
            if (node_2 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_2, node_0))
            if (node_3 < n) or (node_0 < n):
                tsslltd_core_pb_edges.append((node_3, node_0))
            if (node_3 < n) or (node_1 < n):
                tsslltd_core_pb_edges.append((node_3, node_1))
            if (node_3 < n) or (node_2 < n):
                tsslltd_core_pb_edges.append((node_3, node_2))
            else: pass
    
    del simplex, simplices, tsslltd_core_delaunay

    # Convert edge list to np.ndarray, and retain the unique edges from
    # the core and periodic boundary nodes
    tsslltd_core_pb_edges = unique_sorted_edges(tsslltd_core_pb_edges)

    # Lists for the edges of the graph capturing the periodic
    # connections between the core nodes
    conn_core_edges = []
    conn_pb_edges = []

    for edge in range(np.shape(tsslltd_core_pb_edges)[0]):
        node_0 = int(tsslltd_core_pb_edges[edge, 0])
        node_1 = int(tsslltd_core_pb_edges[edge, 1])

        # Edge is a core edge
        if (node_0 < n) and (node_1 < n):
            conn_core_edges.append((node_0, node_1))
        # Edge is a periodic boundary edge
        else:
            node_0 = int(pb2core_nodes[node_0])
            node_1 = int(pb2core_nodes[node_1])
            conn_pb_edges.append((node_0, node_1))
    
    # Convert edge lists to np.ndarrays, and retain unique edges
    conn_core_edges = unique_sorted_edges(conn_core_edges)
    conn_pb_edges = unique_sorted_edges(conn_pb_edges)

    # Save fundamental graph constituents from this topology
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")
    
def delaunay_network_topology(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        n: int,
        config: int) -> None:
    """Delaunay-triangulated network topology.

    This function confirms that the network being called for is a
    Delaunay-triangulated network. Then, the function calls the
    Delaunay-triangulated network initialization function to create the
    Delaunay-triangulated network.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        n (int): Number of core nodes.
        config (int): Configuration number.
    
    """
    # Network topology initialization procedure is only applicable for 
    # Delaunay-triangulated networks. Exit if a different type of
    # network is passed.
    if network != "delaunay":
        import sys
        
        error_str = (
            "Network topology initialization procedure is only "
            + "applicable for Delaunay-triangulated networks. This "
            + "procedure will only proceed if network = ``delaunay''."
        )
        sys.exit(error_str)
    delaunay_network_topology_initialization(
        network, date, batch, sample, scheme, dim, n, config)

def delaunay_network_additional_node_seeding(
        network: str,
        date: str,
        batch: str,
        sample: int,
        scheme: str,
        dim: int,
        b: float,
        n: int,
        config: int,
        max_try: int) -> None:
    """Additional node placement procedure for Delaunay-triangulated
    networks.

    This function generates necessary filenames and calls upon a
    corresponding helper function to calculate and place additional
    nodes in the simulation box.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        scheme (str): Lower-case acronym indicating the particular scheme used to generate the positions of the core nodes; either "random", "prhd", "pdhu", or "lammps" (corresponding to the random node placement procedure ("random"), periodic random hard disk node placement procedure ("prhd"), periodic disordered hyperuniform node placement procedure ("pdhu"), or nodes randomly placed and minimized via LAMMPS ("lammps")).
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Node diameter.
        n (int): Intended total number of nodes.
        config (int): Configuration number.
        max_try (int): Maximum number of node placement attempts for the periodic random hard disk node placement procedure ("prhd").
    
    """
    # The Delaunay-triangulated network additional node placement
    # procedure is only applicable for Delaunay-triangulated networks.
    # Exit if a different type of network is passed.
    if network != "delaunay":
        import sys

        error_str = (
            "Delaunay-triangulated network additional node placement "
            + "procedure is only applicable for Delaunay-triangulated "
            + "networks. This procedure will only proceed if "
            + "network = ``delaunay''."
        )
        sys.exit(error_str)

    # Generate filenames
    L_filename = L_filename_str(network, date, batch, sample)
    coords_filename = (
        delaunay_filename_str(network, date, batch, sample, config) + ".coords"
    )

    # Call additional node placement helper function
    additional_node_seeding(
        L_filename, coords_filename, scheme, dim, b, n, max_try)

def delaunay_network_hilbert_node_label_assignment(
        network: str,
        date: str,
        batch: str,
        sample: int,
        config: int) -> None:
    """Node label assignment procedure for Delaunay-triangulated
    networks.

    This function assigns numerical labels to nodes in
    Delaunay-triangulated networks based on the Hilbert space-filling
    curve. The node coordinates and network edges are updated
    accordingly.

    Args:
        network (str): Lower-case acronym indicating the particular type of network that is being represented by the eventual network topology; here, only "delaunay" is applicable (corresponding to Delaunay-triangulated networks ("delaunay")).
        date (str): "YYYYMMDD" string indicating the date during which the network batch and sample data was generated.
        batch (str): Single capitalized letter (e.g., A, B, C, ...) indicating the batch label of the network sample data.
        sample (int): Label of a particular network in the batch.
        config (int): Configuration number.
    
    """
    # The Delaunay-triangulated network node label assignment procedure
    # is only applicable for Delaunay-triangulated networks. Exit if a
    # different type of network is passed.
    if network != "delaunay":
        import sys

        error_str = (
            "Delaunay-triangulated network node label assignment "
            + "procedure is only applicable for Delaunay-triangulated "
            + "networks. This procedure will only proceed if "
            + "network = ``delaunay''."
        )
        sys.exit(error_str)

    # Load L
    L = np.loadtxt(L_filename_str(network, date, batch, sample))

    # Generate filenames
    delaunay_filename = delaunay_filename_str(
        network, date, batch, sample, config)
    coords_filename = delaunay_filename + ".coords"
    conn_core_edges_filename = delaunay_filename + "-conn_core_edges.dat"
    conn_pb_edges_filename = delaunay_filename + "-conn_pb_edges.dat"

    # Load node coordinates
    coords = np.loadtxt(coords_filename)
    dim = np.shape(coords)[1]

    # Load edges
    conn_core_edges = np.loadtxt(conn_core_edges_filename, dtype=int)
    conn_pb_edges = np.loadtxt(conn_pb_edges_filename, dtype=int)
    conn_core_m = np.shape(conn_core_edges)[0]
    conn_pb_m = np.shape(conn_pb_edges)[0]

    # Assign node labels via the Hilbert space-filling curve 
    hilbert_node_labels = hilbert_node_label_assignment(coords, L, dim)

    # Construct an np.ndarray that returns the index for each node
    # number in the hilbert_node_labels np.ndarray
    hilbert_node_labels_indcs = (
        -1 * np.ones(np.max(hilbert_node_labels)+1, dtype=int)
    )
    hilbert_node_labels_indcs[hilbert_node_labels] = np.arange(
        np.shape(hilbert_node_labels)[0], dtype=int)
    
    # Update the node coordinates with the updated node labels
    coords = coords[hilbert_node_labels]

    # Update original node labels with updated node labels
    for edge in range(conn_core_m):
        conn_core_edges[edge, 0] = int(
            hilbert_node_labels_indcs[conn_core_edges[edge, 0]])
        conn_core_edges[edge, 1] = int(
            hilbert_node_labels_indcs[conn_core_edges[edge, 1]])
    for edge in range(conn_pb_m):
        conn_pb_edges[edge, 0] = int(
            hilbert_node_labels_indcs[conn_pb_edges[edge, 0]])
        conn_pb_edges[edge, 1] = int(
            hilbert_node_labels_indcs[conn_pb_edges[edge, 1]])
    
    # Save updated node coordinates
    np.savetxt(coords_filename, coords)

    # Save updated edges
    np.savetxt(conn_core_edges_filename, conn_core_edges, fmt="%d")
    np.savetxt(conn_pb_edges_filename, conn_pb_edges, fmt="%d")