#!/user/bin/env python3
import copy
import random
import numpy as np

import settings
from .optimizer.adagrad import AdaGrad

Epoch   = settings.Epoch
Lambda  = settings.Lambda
epsilon = settings.Epsilon
batch_size    = settings.BatchSize
sample_size   = settings.SampleSize
learning_rate = settings.InitialLearningRate

# sigmoid function
def sigma(z):
    return 1/(1+np.exp(-z))

# the i-th standard_basis in n-dim Euclidian space
def standard_basis(n: int, i: int) -> np.ndarray:
    e = np.zeros(n, dtype=int); e[i] = 1
    return e

# if sample in the same posision as `normal`: +1; else: -1
def two_valued_classifier(sample: np.ndarray, normal: np.ndarray) -> int:
    return 1 if np.dot(normal, sample) > 0 else -1

# TODO: lambda regularization -> normalization of normal vector
class Objective(object):
    def __init__(self, epsilon=1e-8, Lambda=4):
        self.epsilon = epsilon
        self.Lambda = Lambda

    # We assume that:
    ## knn_list: k-nearest neighbours of feature_index in feature_vector_dict
    def value(self, knn_list, feature_index, feature_vector_dict, samples_index, normal):
        val = 0
        feature_vector = feature_vector_dict[feature_index]
        constant = two_valued_classifier(feature_vector, normal)
        for index in knn_list:
            knn_vector = feature_vector_dict[index]
            z = constant*np.dot(knn_vector, normal)
            if -z > 700:
                val += z
            else:
                val += np.log(sigma(z))

        for random_index in samples_index:
            random_vector = feature_vector_dict[random_index]
            z = -constant*np.dot(random_vector, normal)
            if -z > 700:
                val += z
            else:
                val += np.log(sigma(z))

        # regularization
        val -= Lambda * np.linalg.norm(normal, 1)
        return val

    def gradient(self, knn_list, feature_index, feature_vector_dict, samples_index, normal):
        epsilon, Lambda, M = self.epsilon, self.Lambda, normal.size
        return (1/epsilon)*np.array([
            self.value(knn_list, feature_index, feature_vector_dict, samples_index, normal + epsilon * standard_basis(M, i)) \
            - self.value(knn_list, feature_index, feature_vector_dict, samples_index, normal) for i in range(M)
        ])
    
                                                                          
class LearnHyperPlane(object):
    def __init__(self, M: int, knn, feature_vector_dict, inverted_index, init_normal):
        self.M = M # dimension of feature vector space
        self._knn = knn # knn of `KNNG`, k-nearest neighbors' index
        self._feature_vector_dict = feature_vector_dict # {index: feature vector (: np.ndarray)}
        self._inverted_index = inverted_index # InvertedIndex object
        self.normal = init_normal # normal vector of hyperplane

    def learn(self, debug=False):
        knn = self._knn
        inverted_index = self._inverted_index
        feature_vector_dict = self._feature_vector_dict
        feature_index_set = set(feature_vector_dict.keys())
        feature_index_list = list(feature_vector_dict.keys())

        # AdaGrad
        objective = Objective(epsilon=epsilon, Lambda=Lambda)
        optimizer = AdaGrad(learning_rate=learning_rate)
        
        for epoch in range(Epoch):
            # mini batch learning
            samples_index = random.sample(feature_vector_dict.keys(), sample_size)
            batch_index = random.sample(feature_index_list, batch_size)
            params = {"normal": self.normal}
            grads = {"normal": np.zeros_like(self.normal)}
            
            for i in batch_index:
                index_list = list(set(inverted_index.get(i))&feature_index_set)
                knn_list = knn.get_index(feature_vector_dict[i], index_list)
                feature_index_list.remove(i)
                grads["normal"] -= (1/batch_size)*objective.gradient(knn_list, i, feature_vector_dict, samples_index, self.normal)

            optimizer.update(params, grads)
            if debug:
                print("Epoch: {}, Value: {}".format(
                    epoch,
                    objective.value(knn_list, i, feature_vector_dict, samples_index, self.normal)
                ))
