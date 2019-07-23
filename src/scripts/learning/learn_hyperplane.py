#!/user/bin/env python3
import copy
import random
import numpy as np

from optimizer.sgd import SGD

# sigmoid function
def sigma(z):
    return 1/(1+np.exp(-z))

# L_p norm
def norm(x: np.ndarray, p: int):
    return (x**p).sum()**(1/p)

# the i-th standard_basis in n-dim Euclidian space
def standard_basis(n: int, i: int) -> np.ndarray:
    e = np.zeros(n, dtype=int); e[i] = 1
    return e

# if sample in the same posision as `normal`: +1; else: -1
def two_valued_classifier(sample: np.ndarray, normal: np.ndarray) -> int:
    return 1 if np.dot(normal, sample) > 0 else -1

# TODO: lambda regularization -> normalization of normal vector
# objective function
def E(graph, feature_index: int, feature_vector: np.ndarray, samples_index: list, normal) -> np.float64:
    value = 0
    c = two_valued_classifier(feature_vector, normal)
    for v in graph.edges[feature_index]:
        z = c*np.dot(feature_vector, normal)
        if -z > 700:
            value += z
        else:
            value += np.log(sigma(z))

    for j in samples_index:
        z = -c*np.dot(feature_vector, normal)
        if -z > 700:
            value += z
        else:
            value += np.log(sigma(z))

    value += -Lambda*norm(normal, 1)
    return value

def gradient(graph, feature_index, feature_vector, samples_index, normal) -> np.ndarray:
    M = normal.size
    return (1/epsilon)*np.array([
        E(
            graph,
            feature_index,
            feature_vector,
            samples_index,
            normal+epsilon*standard_basis(M, i)
        ) - E(
            graph,
            feature_index,
            feature_vector,
            samples_index,
            normal
        ) for i in range(M)
    ])

class LearnHyperPlane(object):
    def __init__(self, M: int, graph, feature_vector_dict, init_normal):
        self.M = M # dimension of feature vector space
        self._graph = graph # OrientedGraph object (KNNG)
        self._feature_vector_dict = feature_vector_dict # {index: feature vector (: np.ndarray)}
        self.normal = init_normal # normal vector of hyperplane

    def learn(self, debug=False):
        h, eta = epsilon, initial_eta # parameters in AdaGrad

        graph, feature_vector_dict = self._graph, self._feature_vector_dict
        feature_index_list = list(feature_vector_dict.keys())
        n = len(feature_index_list)//batch_size

        # SGD
        for epoch in range(Epoch):
            # an epoch
            samples_index = random.sample(list(feature_vector_dict.keys()), sample_size)
            copied_index_list = copy.copy(feature_index_list)
            for step in range(n):
                batch_index = random.sample(copied_index_list, batch_size)
                for j in batch_index:
                    copied_index_list.remove(j)
            
                grad = (1/batch_size)*np.sum(
                    np.array([gradient(graph, i, feature_vector_dict[i], samples_index, self.normal)\
                              for i in batch_index]),
                    axis=0
                )
                h += norm(grad, 2)**2
                eta = initial_eta/np.sqrt(h)
                self.normal += eta*grad
