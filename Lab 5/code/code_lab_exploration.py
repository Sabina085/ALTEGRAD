"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import collections


############## Task 1

##################
# your code here #

path = '../datasets/CA-HepTh.txt'
G = nx.read_edgelist(path, comments='#', delimiter='\t')
nr_nodes = G.number_of_nodes()
nr_edges = G.number_of_edges()
print('Number of nodes from graph G: ', nr_nodes)
print('Number of edges from graph G: ', nr_edges)

##################



############## Task 2

##################
# your code here #


#number of connected components
# Option 1
nr_connect_comp = sum(1 for _ in nx.connected_components(G))
print('Number of connected components: ',nx.algorithms.components.number_connected_components(G))
# Option 2
print('Number of connected components: ', nr_connect_comp)


if not nx.is_connected(G):
	# the largest connected component
	largest_cc = max(nx.connected_components(G), key=len)
	nr_nodes_largest_cc = len(largest_cc)
	subgraph_largest_cc = G.subgraph(list(largest_cc))
	nr_edges_largest_cc = subgraph_largest_cc.number_of_edges()
	print('Number of nodes from the largest connected component: ', nr_nodes_largest_cc)
	print('Number of edges from the largest connected component: ', nr_edges_largest_cc)
	print('Fraction nodes: ', nr_nodes_largest_cc / nr_nodes)
	print('Fraction edges: ', nr_edges_largest_cc / nr_edges)


##################


############## Task 3
# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

##################
# your code here #
min_degree = np.min(degree_sequence)
print('Minimum degree: ', min_degree)

max_degree = np.max(degree_sequence)
print('Maximum degree: ', max_degree)

mean_degree = np.mean(degree_sequence)
print('Mean degree: ', mean_degree)

##################


############## Task 4

##################
# your code here #

degreeCount = collections.Counter(degree_sequence)
deg, cnt = zip(*degreeCount.items())

fig1, ax1 = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='r')

plt.title("Degree Histogram")
plt.ylabel("Frequency")
plt.xlabel("Degree")

ax1.plot()
fig1.savefig("degree_histogram.pdf")

fig2, ax2 = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='g')

plt.title("log-log Degree Histogram")
plt.ylabel("Frequency")
plt.xlabel("Degree")

ax2.set_yscale('log')
ax2.set_xscale('log')

fig2.savefig("degree_histogram_log.pdf")

plt.show()

##################
