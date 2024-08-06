import networkx as nx
from louvain_numba import best_partition

G = nx.karate_club_graph()
partition = best_partition(G, n_clusters=(2, 10))
print(len(set(partition.values())))
