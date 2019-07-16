#!/user/bin/env python3
import random
import settings
import numpy as np

M = settings.FeatureSpaceDimension
S = settings.RandomSampleNumber
N = settings.EpochNumber
Lambda = settings.Lambda

# if sample in the same posision as normal: +1; else: -1
def two_valued_classifier(sample: np.ndarray, normal) -> int:
        return 1 if np.dot(normal, sample) > 0 else -1

# sigmoid function
def sigma(z):
    return 1/(1+np.exp(-z))

# L_p norm
def norm(x, p):
    return (x**p).sum()**(1/p)

# objective function
def objective_function(graph, data_set, samples_index, normal):
    value = 0
    for i in range(len(data_set)):
        c = two_valued_classifier(data_set[i]["feature"], normal)
        for j in graph.edges[i]:
            z = c*np.dot(data_set[j]["feature"], normal)
            if -z > 600:
                value += z
            else:
                value += np.log(sigma(z))

        for j in samples_index:
            z = -c*np.dot(data_set[j]["feature"], normal)
            if -z > 600:
                value += z
            else:
                value += np.log(sigma(z))
            
        value += -Lambda*norm(normal, 1)

    return value
    
class LearnHyperPlane(object):
    def __init__(self, graph, data_set):
        self._graph = graph # OrientedGraph object
        self._data_set = data_set # list onject
        self.normal = np.random.normal(0, 0.4, M)

    def learn(self):
        for step in range(N):
            
            
