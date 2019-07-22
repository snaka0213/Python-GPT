#!/user/bin/env python3
import settings
import numpy as np
from .knng import InvertedIndex, KNNG
from .learn_hyperplane import LearnHyperPlane

MaxInLeaf = settings.MaxInLeaf
k = settings.KNNGNumber

def two_valued_classifier(sample: np.ndarray, normal: np.ndarray) -> bool:
    return np.dot(normal, sample) > 0

class Node(object):
    def __init__(self, left=None, right=None, normal=None, label=None):
        self.left = left # `left tree`
        self.right = right # `right tree`
        self.normal = normal # np.ndarray object
        self.label = label

    def isleaf(self) -> bool:
        return self.left is None and self.right is None


class ClassificationTree(object):
    def __init__(self, L: int, M: int, inverted_index: InvertedIndex):
        self.L = L
        self.M = M
        self._root = Node()
        self._index = inverted_index

    def load(self, data_set: dict):
        init_normal = np.random.normal(0, 0.4, self.M)
        self._root = self._grow_tree(data_set, init_normal)

    def _grow_tree(self, data_set: dict, init_normal) -> Node:
        if len(data_set) <= MaxInLeaf:
            label = self._empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._split_node(data_set, init_normal))

    def _empirical_label_distribution(self, data_set): # -> label vector, 1xL np.ndarray object
        if data_set:
            return (1/len(data_set))*np.sum(np.array([data_set[key]["label"] for key in data_set]), axis=0)
        else:
            return np.zeros(self.L, dtype=int)

    def _split_node(self, data_set: dict, init_normal) -> tuple:
        L, M = self.L, self.M
        feature_vector_dict = {key: data_set[key]["feature"] for key in data_set}
        
        knng = KNNG(k, L, data_set, self._index)
        graph = knng.get_graph()
        lhp = LearnHyperPlane(M, graph, feature_vector_dict, init_normal)

        ### Learning Part ###
        lhp.learn(settings.DEBUG)

        normal = lhp.normal
        left, right = {}, {}
        for key in data_set:
            if two_valued_classifier(data_set[key]["feature"], normal):
                left[key] = data_set[key]
            else:
                right[key] = data_set[key]

        left_tree = self._grow_tree(left, normal)
        right_tree = self._grow_tree(right, normal)
        return (left_tree, right_tree, normal)

    def classify(self, sample: np.ndarray): # -> label vector, 1xL np.ndarray object
        pointer = self._root
        while not pointer.isleaf():
            normal = pointer.normal
            if two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer.label

