"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans, SpectralClustering


############## Task 5
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    ##################
    # your code here #

    # scipy_adj_matrix = adjacency_matrix(G)
    # adj_matrix = scipy_adj_matrix.toarray() #convert from scipy to numpy array

    scipy_laplacian = nx.laplacian_matrix(G)
    laplacian = scipy_laplacian.toarray(G) 

    eigVals, eigVects = np.linalg.eigh(laplacian)

    d = 3

    min_indexes = np.argsort(eigVals)[:d]

    first_d_eigen_vectors = eigVects[:, min_indexes]

    list_vectors = []

    for i in range(first_d_eigen_vectors.shape[0]):
    	list_vectors.append(first_d_eigen_vectors[i, :])

    kmeans = KMeans(n_clusters = k).fit(list_vectors)

    labels = kmeans.labels_

    clustering = {}
    
    nodes_indexes = list(G.nodes)

    for i in range(len(labels)):
    	clustering[nodes_indexes[i]] = labels[i]


    ##################
    
    return clustering



############## Task 6

##################
# your code here #

path = '../datasets/CA-HepTh.txt'
G = nx.read_edgelist(path, comments = '#', delimiter = '\t')
nr_clusters = 50
if not nx.is_connected(G):
	largest_cc = max(nx.connected_components(G), key=len)
	nr_nodes_largest_cc = len(largest_cc)
	subgraph_largest_cc = G.subgraph(list(largest_cc))

	dict_clustering = spectral_clustering(subgraph_largest_cc, nr_clusters)
	for i in range(nr_clusters):
		print('Cluster ', str(i), ': ', sum(value == i for value in dict_clustering.values()))


##################



############## Task 7
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #

    m = G.number_of_edges()

    nc = len(set(clustering.values()))

    modularity_res = 0.0

    for i in range(nc):
        nodes_component = [k for k,v in clustering.items() if v == i]
        lc = G.subgraph(nodes_component).number_of_edges()
        degree_sequence = [G.degree(node) for node in nodes_component]
        dc = sum(degree_sequence)
        modularity_res += ((lc / m) - ((dc / (2.*m)) * (dc / (2.*m))))
        
    ##################
    
    return modularity_res



############## Task 8

##################
# your code here #

### Modularity for spectral clustering algorithm using k = 50

modularity_res = modularity(subgraph_largest_cc, dict_clustering)
print('Modularity result for spectral clustering: ', modularity_res)

dict_random_clustering = {}

nodes_largest_cc = list(largest_cc)
for i in range(nr_nodes_largest_cc):
	dict_random_clustering[nodes_largest_cc[i]] = randint(0, nr_clusters - 1)

modularity_res_random = modularity(subgraph_largest_cc, dict_random_clustering)
print('Modularity result for random clustering: ', modularity_res_random)

##################
