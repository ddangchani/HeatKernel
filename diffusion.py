import numpy as np
import scipy.sparse as sp
import networkx as nx
import geopandas as gpd
from linnet import discretize_network

# Diffusion intensity estimator

def diffusion(linnet, points, bw, resolution):
    """
    Diffusion intensity estimator for a linear network.

    Parameters
    ----------
    linnet : linnet object
        Linear network object.
    points : GeoDataFrame
        GeoDataFrame containing the points.
    bw : float
        Bandwidth.
    resolution : float
        Resolution of the discretization.

    Returns
    -------
    f : array
        Array of intensity values evaluated at each discretized node.
    """
    linnet = discretize_network(linnet, resolution)
    n_nodes = len(linnet.nodes)
    N = len(points)
    # Anderson-Morley bound
    D = np.array([a[1] for a in list(linnet.graph.degree)]) # degree matrix
    AM = max(np.array([D[u] + D[v] for u,v in linnet.edges[['node_start', 'node_end']].values]))

    # Construct the associated matrix A
    A = linnet.adjacency
    M = A - np.diag(D) # Centered incidence matrix

    # Bound for delta t
    bound_dt1 = 2 * (resolution ** 2) / AM
    bound_dt2 = bw * resolution / 3
    bound_dt = min(bound_dt1, bound_dt2)
    dt = bound_dt / 2

    # alpha and A
    beta = 0.5
    alpha = beta * dt / (resolution ** 2)
    matA = np.eye(n_nodes) + alpha * M
    matA = sp.csr_matrix(matA) # Convert to sparse matrix

    # Initialize the intensity vector
    f = np.zeros(n_nodes)
    for p in points.geometry:
        snap, nedge = linnet.snap(p)
        node_start = nedge.node_start
        node_end = nedge.node_end
        node_len = nedge.mm_len
        p = snap.distance(linnet.nodes.geometry[node_end]) / node_len
        f[node_start] += p
        f[node_end] += 1 - p

    num_steps = int(bw**2 / dt)

    for i in range(num_steps):
        f = matA.dot(f)

    # To Edges
    f_edges = np.zeros(len(linnet.edges))
    for i, edge in enumerate(linnet.edges.itertuples()):
        f_edges[i] = (f[edge.node_start] + f[edge.node_end]) / 2

    res = gpd.GeoDataFrame({'geometry': linnet.edges.geometry, 'intensity': f_edges})

    return res
    
