"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random


############## Task 9
# Generate simple dataset
def create_dataset():
    Gs = list()
    y = list()

    ##################
    # your code here #

    for number_nodes in range(3, 103):
        G_cycle = nx.cycle_graph(number_nodes)
        Gs.append(G_cycle)
        y.append(0)

        G_path = nx.path_graph(number_nodes)
        Gs.append(G_path)
        y.append(1)

    ##################

    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(G_train), len(all_paths)))
    for i in range(len(G_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



############## Task 10
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    ##################
    # your code here #

    for graph_index in range(len(Gs_train)):
        G = Gs_train[graph_index]
        for sample_index in range(n_samples):

            node1, node2, node3 = np.random.choice(list(G.nodes()), size=3, replace=False)

            nodes_3_graph = G.subgraph([node1, node2, node3])

            for graphlet_index in range(4):
                graphlet = graphlets[graphlet_index]
                if nx.is_isomorphic(nodes_3_graph, graphlet):
                    phi_train[graph_index, graphlet_index] += 1 

    ##################


    phi_test = np.zeros((len(G_test), 4))
    
    ##################
    # your code here #

    for graph_index in range(len(Gs_test)):
        G = Gs_test[graph_index]
        for sample_index in range(n_samples):

            node1, node2, node3 = np.random.choice(list(G.nodes()), size=3, replace=False)

            nodes_3_graph = G.subgraph([node1, node2, node3])

            for graphlet_index in range(4):
                graphlet = graphlets[graphlet_index]
                if nx.is_isomorphic(nodes_3_graph, graphlet):
                    phi_test[graph_index, graphlet_index] += 1 


    ##################

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 11

##################
# your code here #

K_train_g, K_test_g = graphlet_kernel(G_train, G_test, n_samples=200)


##################



############## Task 12

##################
# your code here #


#SVM for shortest path

# Initialize SVM an train
clf = SVC(kernel = 'precomputed')
clf.fit(K_train_sp, y_train)
# Predict
y_pred_sp = clf.predict(K_test_sp)

accuracy_sp = accuracy_score(y_test, y_pred_sp, normalize=False)
print(y_test)
print(y_pred_sp)
print('Accuracy shortest path: ', accuracy_sp, ' correct out of ', len(y_test), '(', (accuracy_sp * 100.) / (1. * len(y_test)),'%)')



# SVM for the graphlet kernel

# Initialize SVM an train
clf2 = SVC(kernel = 'precomputed')
clf2.fit(K_train_g, y_train)
# Predict
y_pred_g = clf2.predict(K_test_g)

accuracy_g = accuracy_score(y_test, y_pred_g, normalize=False)
print(y_test)
print(y_pred_g)
print('Accuracy graphlet kernel: ', accuracy_g, ' correct out of ', len(y_test), '(', (accuracy_g * 100.) / (1. * len(y_test)),'%)')

##################