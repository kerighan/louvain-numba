# Louvain-Numba

Louvain-Numba is a Python library for community detection in graphs using the Louvain method with Numba optimization. This library leverages Numba to accelerate the community detection process, making it suitable for large-scale graphs.

## Features

- Efficient community detection using the Louvain method
- Optimized with Numba for high performance
- Supports various graph input types (NetworkX, NumPy arrays, SciPy sparse matrices)
- Generates hierarchical community structures

## Installation

You can install Louvain-Numba via pip:

```bash
pip install louvain-numba
```

Or using Poetry:

```bash
poetry add louvain-numba
```

## Usage

### Basic Usage

```python
import networkx as nx
import louvain_numba as lvn

# Create a random graph
G = nx.random_partition_graph([100, 100, 100], 0.1, 0.01)

# Find the best partition
partition = lvn.best_partition(G)
print("Best partition:", partition)

# Calculate modularity
modularity = lvn.modularity(partition, G)
print("Modularity:", modularity)
```

### Specify prefered number of clusters

```python
import networkx as nx
import louvain_numba as lvn

# Create a random graph
G = nx.random_partition_graph([100, 100, 100], 0.1, 0.01)

# Find the best partition
partition = lvn.best_partition(G, n_clusters=(7, 10))
print("Best partition:", partition)

# Calculate modularity
modularity = lvn.modularity(partition, G)
print("Modularity:", modularity)
```

### Full Hierarchy

```python
import networkx as nx
import louvain_numba as lvn

# Create a random graph
G = nx.random_partition_graph([100, 100, 100], 0.1, 0.01)

# Generate full hierarchy of partitions
for level, (partition, modularity) in enumerate(lvn.find_partitions(G, return_modularity=True)):
    print(f"Level {level}:")
    print("Partition:", partition)
    print("Modularity:", modularity)
```

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request on GitHub.
License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

This library was inspired by the cylouvain package and leverages Numba for performance optimization.