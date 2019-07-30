#!/user/bin/env python3
import copy
import random
import numpy as np

import settings
from .optimizer.sgd import SGD
from .optimizer.adagrad import AdaGrad
from joblib import Parallel, delayed

Epoch   = settings.Epoch
Lambda  = settings.Lambda
Threads = settings.Threads
epsilon = settings.Epsilon
sample_size   = settings.SampleSize
learning_rate = settings.InitialLearningRate

# sigmoid
def sigma(z):
    return 1/(1+np.exp(-z))

# log of sigmoid
def log_sigmoid(*args):
    return sum([z if -z > 700 else np.log(sigma(z)) for z in args])

# the i-th standard_basis in n-dim Euclidian space
def standard_basis(n: int, i: int, v: np.float64) -> np.ndarray:
    e = np.zeros(n, dtype=np.float64); e[i] = v
    return e

# if sample in the same posision as `normal`: +1; else: -1
def two_valued_classifier(sample: np.ndarray, normal: np.ndarray) -> int:
    return 1 if sample @ normal.T > 0 else -1

# TODO: lambda regularization -> normalization of normal vector
class Loss(object):
    def __init__(self, epsilon=1e-8, Lambda=4):
        self.epsilon = epsilon
        self.Lambda = Lambda

    # We assume that:
    ## knn_list: k-nearest neighbours of feature_index in feature_vector_dict
    def value(self, knn_list, feature_index, feature_vector_dict, samples_index, normal):
        val = 0
        feature_vector = feature_vector_dict[feature_index]
        constant = two_valued_classifier(feature_vector, normal)

        val += log_sigmoid(*[(constant * feature_vector_dict[index] @ normal.T) for index in knn_list])
        val += log_sigmoid(*[(-constant * feature_vector_dict[random_index] @ normal.T) for random_index in samples_index])

        # regularization
        val -= self.Lambda * np.linalg.norm(normal, 1)
        return -val

    def gradient(self, knn_list, feature_index, feature_vector_dict, samples_index, normal):
        epsilon, Lambda, M = self.epsilon, self.Lambda, normal.size
        return np.array([
            (self.value(knn_list, feature_index, feature_vector_dict, samples_index, normal + standard_basis(M, j, epsilon)) \
            - self.value(knn_list, feature_index, feature_vector_dict, samples_index, normal))*(1/epsilon) for j in range(M)
        ])

class LearnHyperPlane(object):
    def __init__(self, M: int, knn, feature_vector_dict, inverted_index, init_normal, debug):
        self.M = M # dimension of feature vector space
        self._knn = knn # knn of `KNNG`, k-nearest neighbors' index
        self._feature_vector_dict = feature_vector_dict # {index: feature vector (: np.ndarray)}
        self._inverted_index = inverted_index # InvertedIndex object
        self.normal = init_normal # normal vector of hyperplane
        self._debug = debug # print debug

    def learn(self):
        feature_vector_dict = self._feature_vector_dict
        params = {"normal": self.normal}
        Parallel(n_jobs=Threads, verbose=5)(delayed(self.job)(i, params) for i in feature_vector_dict.keys())

    # epochs
    def job(self, i, params):
        feature_vector_dict = self._feature_vector_dict
        samples_index = random.sample(feature_vector_dict.keys(), sample_size)

        # AdaGrad
        optimizer = AdaGrad(learning_rate=learning_rate)
        for epoch in range(Epoch):
            self.learn_in_epoch(i, params, samples_index, optimizer)
            
        if self._debug:
            print("### Learning at index {}: Done. ###".format(i))
                

    # an epoch
    def learn_in_epoch(self, i, params, samples_index, optimizer):
        M = self.M
        knn = self._knn
        inverted_index = self._inverted_index
        feature_vector_dict = self._feature_vector_dict

        knn_list = knn.get_index(feature_vector_dict[i], list(set(inverted_index.get(i))&set(feature_vector_dict.keys())))
        normal = params["normal"]

        # refactored version
        constant = two_valued_classifier(feature_vector_dict[i], normal)
        normal_clness = {p: constant*feature_vector_dict[p]@normal.T for p in knn_list}
        sample_clness = {p: -constant*feature_vector_dict[p]@normal.T for p in samples_index}

        loss_at_point = -log_sigmoid(*[normal_clness[p] for p in knn_list])-log_sigmoid(*[sample_clness[p] for p in samples_index])
        if self._debug:
            value = loss_at_point
            print("Index: {}, Loss: {}".format(i, value))

        norm_diff_along = [
            np.abs(normal[q]+epsilon)-np.abs(normal[q]) for q in range(M)
        ]
        loss_perturbed_along = [
            -log_sigmoid(*[normal_clness[p]+epsilon*constant*feature_vector_dict[p][q] for p in knn_list]) \
            -log_sigmoid(*[sample_clness[p]-epsilon*constant*feature_vector_dict[p][q] for p in samples_index]) \
            +Lambda*norm_diff_along[q] for q in range(M)
        ]

        grads = {"normal": np.array([(1/epsilon)*(loss_perturbed_along[q]-loss_at_point) for q in range(M)])}
        optimizer.update(params, grads)
