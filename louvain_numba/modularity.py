import numpy as np
from numba import jit, int32, float64
from numba.typed import Dict
from typing import Union

from .typing import FloatArray, IntArray, SparseMatrix, Graph
from .utils import _get_adj_matrix


@jit(nopython=True, fastmath=True, cache=True, nogil=True)
def _compute_modularity(
    partition: IntArray,
    data: FloatArray,
    indptr: IntArray,
    indices: IntArray, resolution: float
) -> float:
    """Compute the modularity of the given partition."""
    n_nodes = indptr.shape[0] - 1
    part = np.zeros(n_nodes, dtype=int32)
    for node in range(n_nodes):
        part[node] = partition[node]

    links = 0.0
    degrees = np.zeros(n_nodes, dtype=float64)
    for node in range(n_nodes):
        degree = 0.0
        for i in range(indptr[node], indptr[node + 1]):
            if indices[i] == node:
                degree += 2 * data[i]
                links += 2 * data[i]
            else:
                degree += data[i]
                links += data[i]
        degrees[node] = degree
    links /= 2.0

    inc = Dict.empty(key_type=int32, value_type=float64)
    deg = Dict.empty(key_type=int32, value_type=float64)

    for node in range(n_nodes):
        com = part[node]
        if com not in deg:
            deg[com] = 0.0
        deg[com] += degrees[node]
        for i in range(indptr[node], indptr[node + 1]):
            neighbor = indices[i]
            edge_weight = data[i]
            if part[neighbor] == com:
                if com not in inc:
                    inc[com] = 0.0
                if neighbor == node:
                    inc[com] += edge_weight
                else:
                    inc[com] += edge_weight / 2.0

    res = 0.0
    for com in deg:
        res += (resolution * (inc[com] / links) -
                (deg[com] / (2.0 * links)) ** 2)

    return res


def modularity(
    partition: Union[dict, IntArray],
    graph: Union[SparseMatrix, np.ndarray, Graph],
    resolution: float = 1.0
) -> float:
    """Calculate the modularity of a partition."""
    if isinstance(partition, dict):
        partition = np.array(
            [partition[i] for i in range(len(partition))], dtype=int32)
    data, indptr, indices, n_nodes = _get_adj_matrix(graph)
    return _compute_modularity(partition, data, indptr, indices, resolution)
