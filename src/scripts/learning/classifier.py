#!/user/bin/env python3
import random
import numpy as np

from .knng import InvertedIndex, KNNG
from .learn_hyperplane import LearnHyperPlane

class Node(object):
    def __init__(self, left=None, right=None, normal=None, label=None):
        self.left = left # `left tree`
        self.right = right # `right tree`
        self.normal = normal # positive normal vector of hyper plane, np.ndarray object
        self.label = label # label vector, np.ndarray object

    def isleaf(self) -> bool:
        return self.left is None and self.right is None


class ClassificationTree(object):
    def __init__(self, L: int, M: int, k: int, max_in_leaf: int, inverted_index: InvertedIndex, debug=False):
        self.L = L # label vector space dimension
        self.M = M # feature vector space dimension
        self.k = k # k in `KNNG`
        self.MaxInLeaf = max_in_leaf # max size of leafs in output tree
        self._root = Node() # root node
        self._index = inverted_index
        self._debug = debug

    def load(self, data_set: dict):
        random_index = random.choice(data_set.keys())
        init_normal = data_set[random_index]["feature"]
        self._root = self._grow_tree(data_set, init_normal)

    def _grow_tree(self, data_set: dict, init_normal: np.ndarray) -> Node:
        if len(data_set) <= self.MaxInLeaf:
            label = self._empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._split_node(data_set, init_normal))

    def _two_valued_classifier(self, sample: np.ndarray, normal: np.ndarray) -> bool:
        return np.dot(normal, sample) > 0

    def _empirical_label_distribution(self, data_set): # -> label vector, np.ndarray object
        if data_set:
            return (1/len(data_set))*np.sum(np.array([data_set[key]["label"] for key in data_set.keys()]), axis=0)
        else:
            return np.zeros(self.L, dtype=int)

    def _split_node(self, data_set: dict, init_normal: np.ndarray) -> tuple:
        L, M = self.L, self.M
        label_vector_dict = {key: data_set[key]["label"] for key in data_set.keys()}
        feature_vector_dict = {key: data_set[key]["feature"] for key in data_set.keys()}
        
        knng = KNNG(k, L, label_vector_list, self._index)
        lhp = LearnHyperPlane(M, knng.get_graph(), feature_vector_dict, init_normal)

        ### Learning Part ###
        lhp.learn(self._debug)

        normal = lhp.normal
        left, right = {}, {}
        for key in data_set.keys():
            if self._two_valued_classifier(data_set[key]["feature"], normal):
                left[key] = data_set[key]
            else:
                right[key] = data_set[key]

        left_random_index = random.choice(left.keys())
        left_init_normal = data_set[left_random_index]["feature"]
        left_tree = self._grow_tree(left, left_init_normal)

        right_random_index = random.choice(right.keys())
        right_init_normal = data_set[right_random_index]["feature"]
        right_tree = self._grow_tree(right, right_init_normal )
        
        return (left_tree, right_tree, normal)

    def classify(self, sample: np.ndarray): # -> label vector, np.ndarray object
        pointer = self._root
        while not pointer.isleaf():
            normal = pointer.normal
            if two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer.label

