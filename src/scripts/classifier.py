#!/user/bin/env python3
import settings
import numpy as np
from .knng import KNNG
from .learn_hyperplane import LearnHyperPlane

MaxInLeaf = settings.MaxInLeaf
k = settings.KNNGNumber

class Node(object):
    def __init__(self, left=None, right=None, normal=None, label=None):
        self.left = left # `left tree`
        self.right = right # `right tree`
        self.normal = normal # np.ndarray object
        self.label = label

    def isleaf(self) -> bool:
        return self.left is None and self.right is None


class ClassificationTree(object):
    def __init__(self, L: int, M: int, root: Node = Node()):
        self.L = L
        self.M = M
        self._root = root

    '''
    Assume: data_set is a list whose each element has attributes:
    * "feature": feature vector, 1xM np.ndarray object
    *  "label" : label vector, 1xL np.ndarray object
    '''

    def load(self, data_set: list):
        root = self._grow_tree(data_set)
        self._root = root

    def _grow_tree(self, data_set: list) -> Node:
        if len(data_set) < MaxInLeaf:
            label = self._empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._split_node(data_set))

    def _empirical_label_distribution(self, data_set): # -> label vector, 1xL np.ndarray object
        if data_set:
            return (1/len(data_set))*np.sum(np.array([data["label"] for data in data_set]), axis=0)
        else:
            return np.zeros(self.L, dtype=int)

    def _split_node(self, data_set: list) -> tuple:
        L, M = self.L, self.M
        feature_vector_list = [data["feature"] for data in data_set]
        label_vector_list = [data["label"] for data in data_set]
        knng = KNNG(k, L, label_vector_list, approximate=True)
        lhp = LearnHyperPlane(M, knng.graph(), feature_vector_list)

        ### Learning Part ###
        lhp.learn()

        normal = lhp.normal
        left, right = [], []
        for data in data_set:
            if self._two_valued_classifier(data["feature"], normal):
                left.append(data)
            else:
                right.append(data)

        left_tree, right_tree = self._grow_tree(left), self._grow_tree(right)
        return (left_tree, right_tree, normal)

    def classify(self, sample: np.ndarray): # -> label vector, 1xL np.ndarray object
        pointer = self._root
        while not pointer.isleaf():
            normal = pointer.normal
            if self._two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer.label

    def _two_valued_classifier(self, sample: np.ndarray, normal) -> bool:
        return np.dot(normal, sample) > 0
