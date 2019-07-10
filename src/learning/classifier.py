#!/user/bin/env python3
import settings
import numpy as np
from .knng import ConstructApproximateKNNG as knng
from .hyper import LearnHyperPlane as hyper

MaxInLeaf = settings.MaxInLeaf

class Node(object):
    def __init__(self, left=None, right=None, normal=None, label=None):
        self._left = left # `left tree`
        self._right = right # `right tree`
        self._normal = normal # np.ndarray object
        self._label = label

    def isleaf(self) -> bool:
        return self._left is None and self._right is None

    
class ClassificationTree(object):
    def __init__(self, data_set: list):
        root = self._grow_tree(data_set)
        self._root = root

    def _grow_tree(self, data_set: list) -> Node:
        def empirical_label_distribution(data_set): # -> label
            n = len(data_set) if data_set else 1
            return sum([data["label"] for data in data_set)/n

        if len(data_set) < MaxInLeaf:
            label = empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._sprit_node(data_set))

    def _split_node(self, data_set: list) -> tuple:
        g = knng(data_set)
        normal = hyper(g, data_set)
        left, right = [], []
        for data in data_set:
            if self._two_valued_classifier(data["feature"], normal):
                left.append(sample)
            else:
                right.append(sample)

        return (left, right, normal)

    def classify(self, sample: np.ndarray): # -> label
        pointer = self._root
        while not pointer.isleaf():
            normal = pointer._normal
            if self._two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer._label

    def _two_valued_classifier(self, sample: np.ndarray, normal) -> bool:
        return np.dot(normal, sample) > 0
                
