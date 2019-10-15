#!/user/bin/env python3
import time
import random
import numpy as np

import settings
from .knng import KNNG
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

Epoch   = settings.Epoch
Lambda  = settings.Lambda
Threads = settings.Threads
epsilon = settings.Epsilon
sample_size   = settings.SampleSize
learning_rate = settings.InitialLearningRate

# sigmoid
def sigma(z):
    return 1/(1+np.exp(-z)) if -z < 700 else 1

# log of sigmoid
def log_sigmoid(z):
    return np.log(sigma(z))

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
    d = {x: v[x] for x in range(m) if v[x] != 0}
    return d

class Loss(object):
    def __init__(self, epsilon=1e-8, Lambda=4):
        self.epsilon = epsilon
        self.Lambda = Lambda

    def value(self, *, i: int, M: int, G: KNNG,
        data_set: dict, sampling_index: list, w: dict) -> np.float:

        N_i = G.edges[i]
        val = 0
        for j in N_i:
            x = data_set[j]['feature']
            e = two_valued_classifier(x, w)
            z = sum(x[coordinate]*w[coordinate] for coordinate in x.keys() & w.keys())
            val -= log_sigmoid(e*z)

        for j in sampling_index:
            x = data_set[j]['feature']
            e = two_valued_classifier(x, w)
            z = sum(x[coordinate]*w[coordinate] for coordinate in x.keys() & w.keys())
            val -= log_sigmoid(-e*z)

        return val

    def gradient(self, *, i: int, M: int, G: KNNG,
        data_set: dict, sampling_index: list, w: dict) -> np.ndarray:

        N_i = G.edges[i]
        grad = {}
        for j in N_i:
            x = data_set[j]['feature']
            e = two_valued_classifier(x, w)
            z = e*sum(x[coordinate]*w[coordinate] for coordinate in x.keys() & w.keys())

            for coordinate in x.keys():
                if coordinate not in grad.keys():
                    grad[coordinate] = -sigma(-z)*x[coordinate]
                else:
                    grad[coordinate] += -sigma(-z)*x[coordinate]


        for j in sampling_index:
            x = data_set[j]['feature']
            e = two_valued_classifier(x, w)
            z = e*sum(x[coordinate]*w[coordinate] for coordinate in x.keys() & w.keys())

            for coordinate in x.keys():
                if coordinate not in grad.keys():
                    grad[coordinate] = sigma(z)*x[coordinate]
                else:
                    grad[coordinate] += sigma(z)*x[coordinate]

        return grad

class LearnHyperPlane(object):
    def __init__(self, *, M: int, G: KNNG, data_set: dict):
        self.N = len(data_set) # the size of data_set
        self.M = M # feature vector space dimension
        self.G = G # k-nearest neighbor graph
        self._data_set = data_set # dictionary object, {index: data}
        self.normal = {} # dictionary object, {coordinate: value}

        # initialize normal vector
        random_index = random.choice(list(data_set.keys()))
        random_vector = data_set[random_index]['feature'] # dict object

        for coordinate in random_vector.keys():
            self.normal[coordinate] = random_vector[coordinate]

    def learn(self):
        optimizer = AdaGrad(learning_rate=learning_rate)
        print("### Data size: {} ###".format(self.N))

        # estimate time
        checker = random.choice(list(self._data_set.keys()))
        start = time.time()
        self.learn_in_single(checker, optimizer, update=False)
        total = time.time() - start
        print("Estimated time in 1-epoch: {:.1f} sec.".format(total*self.N))

        # online learning
        for epoch in range(Epoch):
            self.learn_in_epoch(optimizer)
            print("Epoch: {} -> Done.".format(epoch))

    # an epoch
    def learn_in_epoch(self, optimizer):
        shuffled_keys_list = random.sample(self._data_set.keys(), self.N)
        for i in shuffled_keys_list:
            self.normal = self.learn_in_single(i, optimizer)

    def learn_in_single(self, i: int, optimizer, update=True) -> dict:
        N = self.N
        M = self.M
        G = self.G
        k = len(G.edges[i])
        data_set = self._data_set
        S = random.sample(data_set.keys(), k)

        loss = Loss(epsilon, Lambda)
        original_grad = loss.gradient(
            i=i, M=M, G=G, data_set=data_set, sampling_index=S, w=self.normal
        )
        for key in self.normal.keys():
            if original_grad.get(key) is not None:
                original_grad[key] += loss.Lambda * np.sign(self.normal[key])
            else:
                original_grad[key] = loss.Lambda * np.sign(self.normal[key])

        params = {'normal': dict_to_vec(M, self.normal)}
        grads  = {'normal': dict_to_vec(M, original_grad)}

        if update:
            optimizer.update(params, grads)

        return vec_to_dict(M, params['normal'])
