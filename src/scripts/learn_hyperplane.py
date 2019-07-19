#!/user/bin/env python3
import numpy as np
import random
import settings

batch_size = settings.BatchSizeInAdaGrad
Epoch = settings.Epoch
Lambda = settings.Lambda
epsilon = settings.Epsilon
initial_eta = settings.InitialLearningRate

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
def E(graph, feature_vector_list, samples_index, normal) -> np.float64:
    N = len(feature_vector_list)
    
    value = 0
    for i in range(N):
        c = two_valued_classifier(feature_vector_list[i], normal)
        for j in graph.edges[i]:
            z = c*np.dot(feature_vector_list[j], normal)
            if -z > 700:
                value += z
            else:
                value += np.log(sigma(z))

        for j in samples_index:
            z = -c*np.dot(feature_vector_list[j], normal)
            if -z > 700:
                value += z
            else:
                value += np.log(sigma(z))

        value += -Lambda*norm(normal, 1)

    return value

def gradient(graph, feature_vector_list, samples_index, normal) -> np.ndarray:
    M = normal.size
    return (1/epsilon)*np.array([
        E(
            graph,
            feature_vector_list,
            samples_index,
            normal+epsilon*standard_basis(M, i)
        ) - E(
            graph,
            feature_vector_list,
            samples_index,
            normal
        ) for i in range(M)
    ])

class LearnHyperPlane(object):
    def __init__(self, M: int, graph, feature_vector_list, init_normal):
        self._graph = graph # OrientedGraph object (KNNG)
        self._feature_vector_list = feature_vector_list # feature vector (: np.ndarray) list
        self.normal = init_normal # normal vector of hyperplane
        self.M = M # dimension of feature vector space


    '''
    # Adam: https://arxiv.org/pdf/1412.6980.pdf
    def learn(self):
        moment_1, moment_2 = np.zeros(self.M), np.zeros(self.M) # momentum in Adam
        epsilon_hat = epsilon*np.sqrt(1-beta_2) # constant in Adam
        
        graph, feature_vector_list = self._graph, self._feature_vector_list
        N = len(feature_vector_list) # the size of data_set
        for epoch in range(Epoch):
            # an epoch
            index_list = list(range(N))
            for step in range(N//batch_size):
                samples_index = random.sample(index_list, batch_size)
                for i in samples_index:
                    index_list.remove(i)
                
                grad = (1/batch_size)*gradient(graph, feature_vector_list, samples_index, self.normal)
                self.normal += eta*grad
        
    '''

    '''
    AdaGrad: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    '''

    def learn(self):
        h, eta = epsilon, initial_eta # parameters in AdaGrad

        graph, feature_vector_list = self._graph, self._feature_vector_list
        N = len(feature_vector_list) # the size of data_set

        for epoch in range(Epoch):
            # an epoch
            index_list = list(range(N))
            for step in range(N//batch_size):
                samples_index = random.sample(index_list, batch_size)
                for i in samples_index:
                    index_list.remove(i)
                
                grad = (1/batch_size)*gradient(graph, feature_vector_list, samples_index, self.normal)
                h += norm(grad, 2)**2
                eta = initial_eta/np.sqrt(h)
                self.normal += eta*grad
    
