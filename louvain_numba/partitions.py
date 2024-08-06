import numpy as np
from numba import jit, float64
from numba.typed import Dict, List
from typing import Union, Tuple, Generator

from .utils import (
    _get_adj_matrix,
    _init_status,
    _neighcom,
    _remove,
    _insert,
    _modularity,
    _renumber,
    _induced_graph,
)
from .typing import FloatArray, IntArray, SparseMatrix, Graph


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _one_step(
    graph_neighbors: List[List[int]],
    graph_weights: List[List[float]],
    node2com: IntArray,
    internals: FloatArray,
    loops: FloatArray,
    degrees: FloatArray,
    gdegrees: FloatArray,
    total_weight: float,
    resolution: float,
) -> None:
    """
    Perform one step of the Louvain method.

    Parameters:
    graph_neighbors (List[List[int]]): Adjacency list of the graph.
    graph_weights (List[List[float]]): Edge weights corresponding
                                       to graph_neighbors.
    node2com (IntArray): Array where the index is the node and the value
                         is the community the node belongs to.
    internals (FloatArray): Array of internal weights for each community.
    loops (FloatArray): Array of loop weights for each node.
    degrees (FloatArray): Array of degrees for each community.
    gdegrees (FloatArray): Array of degrees for each node.
    total_weight (float): Total weight of the graph.
    resolution (float): Resolution parameter for modularity.
    """
    n_nodes = len(graph_neighbors)
    m = 2 * total_weight

    best_increase = -np.inf
    modified = False

    for node in range(n_nodes):
        node_com = node2com[node]
        node_com = np.int64(node_com)

        neighbor_weight = Dict.empty(key_type=np.int64, value_type=float64)
        for i in range(len(graph_neighbors[node])):
            if graph_neighbors[node][i] != node:
                neighborcom = node2com[graph_neighbors[node][i]]
                if neighborcom not in neighbor_weight:
                    neighbor_weight[neighborcom] = graph_weights[node][i]
                else:
                    neighbor_weight[neighborcom] += graph_weights[node][i]

        node_gdegree = gdegrees[node]
        node_loop = loops[node]
        com_weight = neighbor_weight.get(node_com, 0.0)

        degrees[node_com] -= node_gdegree
        internals[node_com] -= com_weight + node_loop
        node2com[node] = -1

        for com in neighbor_weight:
            weight = neighbor_weight[com]
            if weight <= 0:
                continue

            increase = resolution * weight - degrees[com] * node_gdegree / m
            if increase > best_increase:
                best_increase = increase
                best_move_node = node
                best_move_com = com
                best_neighbor_weight = neighbor_weight
                old_move_com = node_com
                modified = True

        node2com[node] = node_com
        degrees[node_com] += node_gdegree
        internals[node_com] += com_weight + node_loop

    if not modified:
        return

    if old_move_com not in degrees:
        degrees[old_move_com] = 0.0
    if old_move_com not in internals:
        internals[old_move_com] = 0.0
    if old_move_com not in gdegrees:
        gdegrees[old_move_com] = 0.0
    if old_move_com not in loops:
        loops[old_move_com] = 0.0

    node2com[best_move_node] = -1
    degrees[old_move_com] -= gdegrees[best_move_node]
    internals[old_move_com] -= (
        best_neighbor_weight.get(old_move_com, 0) + loops[best_move_node]
    )

    node2com[best_move_node] = best_move_com
    degrees[best_move_com] += gdegrees[best_move_node]
    internals[best_move_com] += (
        best_neighbor_weight[best_move_com] + loops[best_move_node]
    )


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _one_level(
    n_nodes: int,
    node2com: IntArray,
    internals: FloatArray,
    loops: FloatArray,
    degrees: FloatArray,
    gdegrees: FloatArray,
    graph_neighbors: List[List[int]],
    graph_weights: List[List[float]],
    total_weight: float,
    resolution: float,
    PASS_MAX: int,
    MIN: float,
) -> None:
    """Perform one level of the Louvain method."""
    modified = True
    nb_pass_done = 0
    cur_mod = _modularity(node2com, internals, degrees, total_weight, resolution)
    new_mod = cur_mod

    while modified and nb_pass_done != PASS_MAX:
        cur_mod = new_mod
        modified = False
        nb_pass_done += 1

        for node in range(n_nodes):
            node_com = node2com[node]
            neighbor_weight = _neighcom(node, graph_neighbors, graph_weights, node2com)
            _remove(
                node,
                node_com,
                neighbor_weight.get(node_com, 0),
                node2com,
                degrees,
                internals,
                loops,
                gdegrees,
            )
            best_com = node_com
            best_increase = 0

            for com in neighbor_weight:
                weight = neighbor_weight[com]
                increase = resolution * weight - (degrees[com] * gdegrees[node]) / (
                    2.0 * total_weight
                )
                if increase > best_increase:
                    best_increase = increase
                    best_com = com

            _insert(
                node,
                best_com,
                neighbor_weight.get(best_com, 0),
                node2com,
                degrees,
                internals,
                loops,
                gdegrees,
            )
            if best_com != node_com:
                modified = True

        new_mod = _modularity(node2com, internals, degrees, total_weight, resolution)
        if new_mod - cur_mod < MIN:
            break


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _generate_dendrogram(
    data: FloatArray,
    indptr: IntArray,
    indices: IntArray,
    n_nodes: int,
    resolution: float = 1.0,
    PASS_MAX: int = -1,
    MIN: float = 1e-7,
    full_hierarchy: bool = False,
) -> Tuple[List[IntArray], List[float]]:
    """Generate the dendrogram using the Louvain method."""
    graph_neighbors = [indices[indptr[i] : indptr[i + 1]] for i in range(n_nodes)]
    graph_weights = [data[indptr[i] : indptr[i + 1]] for i in range(n_nodes)]

    node2com, internals, loops, degrees, gdegrees, total_weight = _init_status(
        n_nodes, graph_neighbors, graph_weights
    )

    _one_level(
        n_nodes,
        node2com,
        internals,
        loops,
        degrees,
        gdegrees,
        graph_neighbors,
        graph_weights,
        total_weight,
        resolution,
        PASS_MAX,
        MIN,
    )
    new_mod = _modularity(node2com, internals, degrees, total_weight, resolution)

    partition_list = []
    modularities = List.empty_list(float64)
    communities, node2com = _renumber(node2com, n_nodes)
    partition_list.append(node2com.copy())
    modularities.append(new_mod)

    new_n_nodes, graph_neighbors, graph_weights = _induced_graph(
        communities, graph_neighbors, graph_weights, node2com
    )
    node2com, internals, loops, degrees, gdegrees, total_weight = _init_status(
        new_n_nodes, graph_neighbors, graph_weights
    )

    while True:
        _one_level(
            new_n_nodes,
            node2com,
            internals,
            loops,
            degrees,
            gdegrees,
            graph_neighbors,
            graph_weights,
            total_weight,
            resolution,
            PASS_MAX,
            MIN,
        )
        mod = new_mod
        new_mod = _modularity(node2com, internals, degrees, total_weight, resolution)
        if new_mod - mod < MIN:
            break
        communities, node2com = _renumber(node2com, new_n_nodes)
        partition_list.append(node2com.copy())
        modularities.append(new_mod)
        new_n_nodes, graph_neighbors, graph_weights = _induced_graph(
            communities, graph_neighbors, graph_weights, node2com
        )
        node2com, internals, loops, degrees, gdegrees, total_weight = _init_status(
            new_n_nodes, graph_neighbors, graph_weights
        )

    if full_hierarchy:
        while True:
            prev_num_communities = len(set(node2com))
            _one_step(
                graph_neighbors,
                graph_weights,
                node2com,
                internals,
                loops,
                degrees,
                gdegrees,
                total_weight,
                resolution,
            )
            new_mod = _modularity(
                node2com, internals, degrees, total_weight, resolution
            )
            communities, node2com = _renumber(node2com, new_n_nodes)
            num_communities = len(set(node2com))
            if num_communities == prev_num_communities:
                break
            else:
                partition_list.append(node2com.copy())
                modularities.append(new_mod)
                new_n_nodes, graph_neighbors, graph_weights = _induced_graph(
                    communities, graph_neighbors, graph_weights, node2com
                )
                node2com, internals, loops, degrees, gdegrees, total_weight = (
                    _init_status(new_n_nodes, graph_neighbors, graph_weights)
                )

    return partition_list, modularities


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _find_best_partition(
    data: FloatArray,
    indptr: IntArray,
    indices: IntArray,
    n_nodes: int,
    resolution: float = 1.0,
) -> Tuple[Dict, float]:
    """Find the best partition of the graph."""
    dendrogram, mods = _generate_dendrogram(
        data, indptr, indices, n_nodes, resolution=resolution
    )

    partition = Dict.empty(key_type=np.int64, value_type=np.int64)
    for i in range(len(dendrogram[-1])):
        partition[i] = i

    for i in range(1, len(dendrogram) + 1):
        new_partition = Dict.empty(key_type=np.int64, value_type=np.int64)
        for j in range(len(dendrogram[-i])):
            new_partition[j] = partition[dendrogram[-i][j]]
        partition = new_partition
    return partition, mods[-1]


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _find_all_partitions(
    data: FloatArray,
    indptr: IntArray,
    indices: IntArray,
    n_nodes: int,
    resolution: float = 1.0,
) -> Generator[Tuple[Dict, float], None, None]:
    """Find all partitions of the graph."""
    dendrogram, mods = _generate_dendrogram(
        data, indptr, indices, n_nodes, full_hierarchy=True, resolution=resolution
    )

    for start_level in range(len(dendrogram), 0, -1):
        partition = Dict.empty(key_type=np.int64, value_type=np.int64)
        for i in range(len(dendrogram[-start_level])):
            partition[i] = i

        for i in range(start_level, len(dendrogram) + 1):
            new_partition = Dict.empty(key_type=np.int64, value_type=np.int64)
            for j in range(len(dendrogram[-i])):
                new_partition[j] = partition[dendrogram[-i][j]]
            partition = new_partition
        yield partition, mods[-start_level]


def find_partitions(
    graph: Union[SparseMatrix, np.ndarray, Graph],
    resolution: float = 1.0,
    return_modularity: bool = False,
) -> Generator[Union[Dict, Tuple[Dict, float]], None, None]:
    """Find all partitions of the given graph."""
    data, indptr, indices, n_nodes = _get_adj_matrix(graph)
    for partition, mod in _find_all_partitions(
        data, indptr, indices, n_nodes, resolution=resolution
    ):
        if return_modularity:
            yield partition, mod
        else:
            yield partition


def best_partition(
    graph: Union[SparseMatrix, np.ndarray, Graph],
    resolution: float = 1.0,
    return_modularity: bool = False,
    n_clusters: Union[tuple, int] = None,
) -> Union[Dict, Tuple[Dict, float]]:
    """Find the best partition of the given graph."""
    data, indptr, indices, n_nodes = _get_adj_matrix(graph)
    if n_clusters is None:
        partition, mod = _find_best_partition(
            data, indptr, indices, n_nodes, resolution=resolution
        )
        if return_modularity:
            return partition, mod
        else:
            return partition
    else:
        # iterate through the partitions and return the one with the desired number of clusters
        best_score = -np.inf
        best_mod = -np.inf
        best_partition = None
        for partition, mod in _find_all_partitions(
            data, indptr, indices, n_nodes, resolution=resolution
        ):
            n_clusters_found = len(set(partition.values()))
            if isinstance(n_clusters, int):
                dist = abs(n_clusters_found - n_clusters)
                score = -dist
            elif isinstance(n_clusters, tuple):
                min_clusters, max_clusters = n_clusters
                if min_clusters <= n_clusters_found <= max_clusters:
                    score = mod * 1e3
            if score > best_score:
                best_mod = mod
                best_score = score
                best_partition = partition
        if return_modularity:
            return best_partition, best_mod
        else:
            return best_partition
