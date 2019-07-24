#!/usr/bin/env python3
import numpy as np

### Adam: https://arxiv.org/pdf/1412.6980.pdf ###
class Adam(object):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, params, grads):
        for key in params.keys():
            params[key] += -self.learning_rate * grads[key]
