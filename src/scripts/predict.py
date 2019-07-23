#!/usr/bin/env python3
import numpy as np
from .learning.file_writer import FileWriter

class Predict(object):
    def __init__(self):
        self.trees = None # tuple of Trees
        self.results = None # sample_list -> label_vector_list

    def load(self, *args):
        self.trees = args

    def predict(self, sample_list: list):
        trees, T = self.trees, len(self.trees)
        self.results = [(1/T)*np.sum([tree.classify(sample) for tree in trees], axis=0) for sample in sample_list]

    def write(self, k: int, file_name: str):
        writer = FileWriter(file_name)
        writer.write(k, self.results)
