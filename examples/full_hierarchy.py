import networkx as nx
from louvain_numba import find_partitions
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
pos = nx.spring_layout(G)
for partition in find_partitions(G):
    colors = [partition[node] for node in G.nodes()]
    nx.draw(G, node_color=colors, with_labels=False, pos=pos)
    plt.show()
