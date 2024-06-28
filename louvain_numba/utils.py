import numpy as np
from numba import jit, int32, float64
from numba.typed import Dict, List
from scipy import sparse
import networkx as nx
from typing import Union, Tuple

from .typing import FloatArray, IntArray, SparseMatrix, Graph


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _init_status(
    n_nodes: int,
    graph_neighbors: List[List[int]],
    graph_weights: List[List[float]]
) -> Tuple[IntArray, FloatArray, FloatArray, FloatArray, FloatArray, float]:
    """Initialize the status of the graph nodes."""
    node2com = np.arange(n_nodes, dtype=int32)
    internals = np.zeros(n_nodes, dtype=float64)
    loops = np.zeros(n_nodes, dtype=float64)
    degrees = np.zeros(n_nodes, dtype=float64)
    gdegrees = np.zeros(n_nodes, dtype=float64)
    total_weight = 0.0

    for node in range(n_nodes):
        for i in range(len(graph_neighbors[node])):
            neighbor = graph_neighbors[node][i]
            neighbor_weight = graph_weights[node][i]
            if neighbor == node:
                internals[node] += neighbor_weight
                loops[node] += neighbor_weight
                degrees[node] += 2.0 * neighbor_weight
                gdegrees[node] += 2.0 * neighbor_weight
                total_weight += 2.0 * neighbor_weight
            else:
                degrees[node] += neighbor_weight
                gdegrees[node] += neighbor_weight
                total_weight += neighbor_weight

    total_weight /= 2.0
    return node2com, internals, loops, degrees, gdegrees, total_weight


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _remove(
    node: int,
    com: int,
    weight: float,
    node2com: IntArray,
    degrees: FloatArray,
    internals: FloatArray,
    loops: FloatArray,
    gdegrees: FloatArray
) -> None:
    """Remove a node from its community."""
    node2com[node] = -1
    degrees[com] -= gdegrees[node]
    internals[com] -= weight + loops[node]


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _insert(
    node: int,
    com: int,
    weight: float,
    node2com: IntArray,
    degrees: FloatArray,
    internals: FloatArray,
    loops: FloatArray,
    gdegrees: FloatArray
) -> None:
    """Insert a node into a community."""
    node2com[node] = com
    degrees[com] += gdegrees[node]
    internals[com] += weight + loops[node]


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _modularity(
    node2com: IntArray,
    internals: FloatArray,
    degrees: FloatArray,
    total_weight: float,
    resolution: float
) -> float:
    """Compute the modularity of the current partition."""
    result = 0.0
    for com in range(node2com.shape[0]):
        if degrees[com] > 0:
            result += resolution * internals[com] / total_weight
            result -= (degrees[com] / (2.0 * total_weight)) ** 2
    return result


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _neighcom(
    node: int,
    graph_neighbors: List[List[int]],
    graph_weights: List[List[float]],
    node2com: IntArray
) -> Dict:
    """Find the neighboring communities of a node."""
    neighbor_weight = Dict.empty(key_type=np.int64, value_type=float64)
    for i in range(len(graph_neighbors[node])):
        if graph_neighbors[node][i] != node:
            neighborcom = node2com[graph_neighbors[node][i]]
            if neighborcom not in neighbor_weight:
                neighbor_weight[neighborcom] = graph_weights[node][i]
            else:
                neighbor_weight[neighborcom] += graph_weights[node][i]
    return neighbor_weight


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _renumber(
    node2com: IntArray,
    n_nodes: int
) -> Tuple[List[List[int]], IntArray]:
    """Renumber the communities for consistency."""
    com_n_nodes = np.zeros(n_nodes, dtype=int32)
    for node in range(n_nodes):
        com_n_nodes[node2com[node]] += 1

    com_new_index = np.zeros(n_nodes, dtype=int32)
    final_index = 0
    for com in range(n_nodes):
        if com_n_nodes[com] > 0:
            com_new_index[com] = final_index
            final_index += 1

    new_communities = List([List.empty_list(int32)
                            for _ in range(final_index)])
    new_node2com = np.zeros(n_nodes, dtype=int32)

    for node in range(n_nodes):
        index = com_new_index[node2com[node]]
        new_communities[index].append(node)
        new_node2com[node] = index

    return new_communities, new_node2com


def _get_adj_matrix(
    graph: Union[SparseMatrix, np.ndarray, Graph]
) -> Tuple[FloatArray, IntArray, IntArray, int]:
    """Get the adjacency matrix components from the input graph."""
    if isinstance(graph, sparse.csr_matrix):
        adj_matrix = graph
    elif isinstance(graph, np.ndarray):
        adj_matrix = sparse.csr_matrix(graph)
    elif isinstance(graph, nx.Graph):
        adj_matrix = nx.adjacency_matrix(graph)
    else:
        raise TypeError(
            "The argument should be a NetworkX graph, a NumPy array or a SciPy"
            " Compressed Sparse Row matrix."
        )

    data, indptr, indices, n_nodes = (
        adj_matrix.data,
        adj_matrix.indptr,
        adj_matrix.indices,
        adj_matrix.shape[0],
    )
    return data, indptr, indices, n_nodes


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _induced_graph(communities, graph_neighbors, graph_weights, node2com):
    """
    Create the induced graph based on the current community assignments.

    Parameters:
    communities (List[List[int]]): List of communities, where each community
                                   is a list of node indices.
    graph_neighbors (List[List[int]]): Adjacency list of the original graph.
    graph_weights (List[List[float]]): Edge weights corresponding to
                                       graph_neighbors.
    node2com (IntArray): Array where the index is the node and the value is
                         the community the node belongs to.

    Returns:
    Tuple[int, List[List[int]], List[List[float]]]: The number of nodes in the
                                                    induced graph, adjacency
                                                    list of the induced graph,
                                                    and edge weights of the
                                                    induced graph.
    """
    new_n_nodes = len(communities)
    new_graph_neighbors = List([List.empty_list(int32)
                                for _ in range(new_n_nodes)])
    new_graph_weights = List([List.empty_list(float64)
                              for _ in range(new_n_nodes)])

    for com in range(new_n_nodes):
        to_insert = Dict.empty(key_type=int32, value_type=float64)
        for node in communities[com]:
            for i in range(len(graph_neighbors[node])):
                neighbor = graph_neighbors[node][i]
                neighbor_com = node2com[neighbor]
                neighbor_weight = graph_weights[node][i]
                if neighbor_com not in to_insert:
                    to_insert[neighbor_com] = 0.0

                if neighbor == node:
                    to_insert[neighbor_com] += 2 * neighbor_weight
                else:
                    to_insert[neighbor_com] += neighbor_weight
        for neighbor_com, weight in to_insert.items():
            new_graph_neighbors[com].append(neighbor_com)
            if neighbor_com == com:
                new_graph_weights[com].append(weight / 2.0)
            else:
                new_graph_weights[com].append(weight)

    return new_n_nodes, new_graph_neighbors, new_graph_weights
