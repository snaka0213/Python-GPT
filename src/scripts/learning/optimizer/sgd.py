#!/user/bin/env python3
import numpy as np

# Stochastic Gradient Descent
class SGD(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] += -self.learning_rate * grads[key]

# Momentum SGD
class MomentumSGD(object):
    def __init__(self, learning_rate=0.01, init_momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = init_momentum
        self.velocity = None

    def update(self, params, grads):
        if self.velocity is None:
            self.velocity = {}
            for key, val in params.items():
                self.velocity[key] = np.zeros_like(val)

        for key in params.keys():
            self.velocity[key] = -self.learning_rate * grads[key] + self.momentum * self.velocity[key]
            params[key] += self.velocity[key]
