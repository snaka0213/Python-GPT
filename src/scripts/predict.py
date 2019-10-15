#!/usr/bin/env python3
import numpy as np

class Predict(object):
    def __init__(self):
        self.trees = None # tuple of Trees

    def load_trees(self, *args):
        self.trees = args

    def predict(self, sample: dict) -> np.ndarray:
        trees, T = self.trees, len(self.trees)
        return (1/T)*np.sum([tree.classify(sample) for tree in trees], axis=0)
