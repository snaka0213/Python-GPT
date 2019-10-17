#!/user/bin/env python3
import numpy as np

### AdaGrad: http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf ###
class AdaGrad(object):
    def __init__(self, learning_rate=0.001, decay=1e-8):
        self.init_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.decay = decay

    def update(self, params, grads):
        for key in grads.keys():
            self.decay += np.sum(grads[key]*grads[key])
        self.learning_rate = self.init_learning_rate / np.sqrt(self.decay)
        for key in params.keys() & grads.keys():
            params[key] += -self.learning_rate * grads[key]
