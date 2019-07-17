#!/usr/bin/env python3
import numpy as np
import settings
from .classifier import ClassificationTree
from .file_writer import FileWriter

k = settings.KOfPrecision

class Predict(object):
    def __init__(self):
        self.trees = None
        self.results = None

    def load(self, *args):
        self.trees = args

    def predict(self, sample_list: list):
        trees = self.trees
        T = len(trees)
        self.results = [(1/T)*np.sum([tree.classify(sample) for tree in trees]) for sample in sample_list]

    def write(self, file_name: str):
        writer = FileWriter(file_name)
        writer.write(k, self.results)
