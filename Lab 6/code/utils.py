"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import scipy.sparse as sp
import numpy as np


def normalize_adjacency(A):
    ############## Task 9
    
    ##################
    # your code here #
    ##################

    A_tilde = A + np.eye(A.shape[0])

    diagonal_vector = np.zeros((A.shape[0], 1))

    for i in range(A.shape[0]):
        sum_col = 0
        for j in range(A.shape[1]):
            sum_col += A[i, j]

        diagonal_vector[i, 0] = sum_col     


    D_tilde = np.diag(np.squeeze(diagonal_vector))
    D_term = D_tilde

    for i in range(D_term.shape[0]):
        D_term[i, i] = 1./np.sqrt(D_term[i, i])

    A_normalized = np.matmul(np.matmul(D_term, A_tilde), D_term)

    return A_normalized


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = features.todense()
    features /= features.sum(1).reshape(-1, 1)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj.todense()

    print('Dataset has {} nodes, {} edges, {} features.'.format(adj.shape[0], edges.shape[0], features.shape[1]))

    return features, adj, labels


def accuracy(output, labels):
    """Computes classification accuracy"""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)