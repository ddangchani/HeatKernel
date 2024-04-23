import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
import momepy
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from itertools import product
import shapely

warnings.filterwarnings("ignore")


class Linnet(nx.MultiGraph):
    """
    Create a networkx MultiGraph object from a GeoDataFrame of LINESTRINGs
    Attributes:
        nodes (GeoDataFrame) : GeoDataFrame of nodes
        edges (GeoDataFrame) : GeoDataFrame of edges
        sw (libpysal.weights.W) : spatial weights object
    """
    def __init__(self, edges):
        super().__init__()
        assert isinstance(edges, (gpd.GeoSeries, gpd.GeoDataFrame)), "Edges must be a GeoSeries or GeoDataFrame object"
        self.graph = momepy.gdf_to_nx(edges)
        nodes, edges, sw = momepy.nx_to_gdf(self.graph, points=True, lines=True, spatial_weights=True)
        self.nodes = nodes
        self.edges = edges
        self._from = self.edges['node_start']
        self._to = self.edges['node_end']
        self._len = self.edges['mm_len']
        self.shortest_path = nx.floyd_warshall_numpy(self.graph, weight='mm_len')
        self.adjacency = nx.adjacency_matrix(self.graph).toarray()

    def snap(self, point):
        """
        Snap a point to the nearest edge on the network
        Args:
            point (Point) : point to snap
        Returns:
            (Point, GeoDataFrame) : snapped point, nearest edge (GeoDataFrame)
        """
        nidx = self.edges.sindex.nearest(point)[1][0]
        nedge = self.edges.loc[nidx]
        snap = nedge.geometry.interpolate(nedge.geometry.project(point))
        return snap, nedge
    


def discretize_network(linnet, resolution):
    """
    Discretize the network into equally spaced points
    Args:
        linnet (Linnet) : Linnet object
        resolution (float) : resolution of equally spaced points along the network
    Returns:
        linnet (Linnet) : Linnet object with discretized network
    """
    # interpolates
    v = int(linnet.edges.unary_union.length / resolution)
    interpolates = gpd.GeoSeries([linnet.edges.unary_union.interpolate(i, normalized=True) for i in np.linspace(0, 1, v)])

    # Discretize the network
    edges = linnet.edges.geometry # GeoSeries
    err = []

    for point in tqdm(interpolates):
        n_index = edges.geometry.sindex.nearest(point)[1][0]
        try:
            new = split_line(edges.geometry[n_index], point,)
            edges.drop(n_index, inplace=True)
            edges = pd.concat([edges, new], ignore_index=True)
        except ValueError:
            err.append(n_index)
    

    edges = gpd.GeoDataFrame(geometry=edges)

    if len(err) > 0:
        print(f"Error in splitting {len(err)} edges")

    return Linnet(edges)

def plot_network(linnet, ax, node_size=15, **kwargs):
    """
    Plot the network and points
    Args:
        linnet (src.linnet.Linnet) : Linnet object
        points (geopandas.GeoDataFrame) : points
        ax (matplotlib.axes._subplots.AxesSubplot) : axis
    """
    nx.draw(linnet.graph, {n : [n[0], n[1]] for n in linnet.graph.nodes()}, ax=ax, node_size=node_size, **kwargs)   
    return ax

def split_line(edge, point):
    edge_start, edge_end = edge.boundary.geoms

    if point == edge_start or point == edge_end:
        return gpd.GeoSeries([edge])

    line1 = shapely.geometry.LineString([edge_start, point])
    line2 = shapely.geometry.LineString([point, edge_end])
    lines = gpd.GeoSeries([line1, line2])

    return lines