#!/user/bin/env python3
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import settings
from .knng import KNNG
from .optimizer.sgd import MomentumSGD
from .optimizer.adam import Adam
from .optimizer.adagrad import AdaGrad

'''
__Assume__
* To handle with sparse vector,
    it seems that list or dict object is better than np.ndarray.

* Given `data_set`, dict object {index: data}
    `data` is dict object, whose keys are:
    *  'label' : `label_vector`, list object subset in [0,...,L-1]
    * 'feature': `feature_vector`, dict object {coordinate index: value}
'''
# set optimizer
opt = AdaGrad

# set hyperparameters
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
def log_sigmoid(z):
    return np.log(sigma(z)) if -z < 700 else z

# if sample in the same posision as `normal`: +1; else: -1
def two_valued_classifier(sample: dict, normal: dict) -> int:
    value = 0
    for key in sample.keys() & normal.keys():
        value += sample[key]*normal[key]

    return 1 if value > 0 else -1

def dict_to_vec(m: int, d: dict):
    vector = np.zeros(m)
    for coordinate in d.keys():
        vector[coordinate] += d[coordinate]

    return vector

def vec_to_dict(m: int, v: np.ndarray):
    return {x: v[x] for x in range(m) if v[x] != 0}

class Loss(object):
    def __init__(self, epsilon=1e-8, Lambda=4):
        self.epsilon = epsilon
        self.Lambda = Lambda

    def value(self, *, i: int, M: int, G: KNNG,
        data_set: dict, sampling_index: list, w: dict) -> np.float:

        N_i = G.edges[i]
        x_i = data_set[i]['feature']
        val = 0
        for j in N_i:
            x_j = data_set[j]['feature']
            e_i = two_valued_classifier(x_i, w)
            z_j = e_i*sum(x_j[coordinate]*w[coordinate] for coordinate in x_j.keys() & w.keys())
            val -= log_sigmoid(z_j)

        for j in sampling_index:
            x_j = data_set[j]['feature']
            e_i = two_valued_classifier(x_i, w)
            z_j = e_i*sum(x_j[coordinate]*w[coordinate] for coordinate in x_j.keys() & w.keys())
            val -= log_sigmoid(-z_j)

        return val

    def gradient(self, *, i: int, M: int, G: KNNG,
        data_set: dict, sampling_index: list, w: dict) -> dict:

        N_i = G.edges[i]
        x_i = data_set[i]['feature']
        grad = {}
        for j in N_i:
            x_j = data_set[j]['feature']
            e_i = two_valued_classifier(x_i, w)
            z_j = e_i*sum(x_j[coordinate]*w[coordinate] for coordinate in x_j.keys() & w.keys())
            s_j = sigma(-z_j) if z_j < 700 else 1

            for coordinate in x_j.keys():
                if coordinate not in grad.keys():
                    grad[coordinate] = -e_i*s_j*x_j[coordinate]
                else:
                    grad[coordinate] += -e_i*s_j*x_j[coordinate]

        for j in sampling_index:
            x_j = data_set[j]['feature']
            e_i = two_valued_classifier(x_i, w)
            z_j = e_i*sum(x_j[coordinate]*w[coordinate] for coordinate in x_j.keys() & w.keys())
            s_j = sigma(z_j) if -z_j < 700 else 1

            for coordinate in x_j.keys():
                if coordinate not in grad.keys():
                    grad[coordinate] = e_i*s_j*x_j[coordinate]
                else:
                    grad[coordinate] += e_i*s_j*x_j[coordinate]

        return grad

class LearnHyperPlane(object):
    def __init__(self, *, M: int, G: KNNG, data_set: dict, init_normal: dict):
        self.N = len(data_set) # the size of data_set
        self.M = M # feature vector space dimension
        self.G = G # k-nearest neighbor graph
        self._data_set = data_set # dictionary object, {index: data}
        self.normal = init_normal # dictionary object, {coordinate: value}

    def learn(self, debug=False):
        loss = Loss(epsilon, Lambda)
        optimizer = opt(learning_rate=learning_rate)
        print("### Data size: {} ###".format(self.N))

        # estimate time
        checker = random.choice(list(self._data_set.keys()))
        start_time = time.time()

        self.learn_in_single(checker, loss, optimizer, update=False)
        total_time = time.time() - start_time
        if debug:
            print("Estimated time in 1-epoch: {:.1f} sec.".format(total_time*self.N))

        # online learning
        for epoch in range(Epoch):
            self.learn_in_epoch(loss, optimizer)
            if debug:
                print("Epoch: {} -> Done.".format(epoch))
        else:
            print("{} epochs learning -> Done.".format(Epoch))

    # an epoch
    def learn_in_epoch(self, loss, optimizer, debug=False):
        shuffled_keys_list = random.sample(self._data_set.keys(), self.N)
        for i in shuffled_keys_list:
            val = self.learn_in_single(i, loss, optimizer, update=True)

    def learn_in_single(self, i: int, loss, optimizer, update=True):
        N = self.N
        M = self.M
        G = self.G
        k = len(G.edges[i])
        data_set = self._data_set
        S = random.sample(data_set.keys(), k)

        grad_vec = loss.gradient(
            i=i, M=M, G=G, data_set=data_set, sampling_index=S, w=self.normal
        )
        for key in self.normal.keys():
            if key not in grad_vec.keys():
                grad_vec[key] = loss.Lambda * np.sign(self.normal[key])
            else:
                grad_vec[key] += loss.Lambda * np.sign(self.normal[key])

        params = {'normal': dict_to_vec(M, self.normal)}
        grads  = {'normal': dict_to_vec(M, grad_vec)}

        if update:
            optimizer.update(params, grads)

        self.normal = vec_to_dict(M, params['normal'])
