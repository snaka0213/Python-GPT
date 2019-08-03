#!/user/bin/env python3
import json
import random
import numpy as np

from .knng import KNN
from .learn_hyperplane import LearnHyperPlane

class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

class Node(object):
    def __init__(self, left=None, right=None, normal=None, label=None):
        self.left = left # `left tree`
        self.right = right # `right tree`
        self.normal = normal # positive normal vector of hyper plane, np.ndarray object
        self.label = label # label vector, np.ndarray object

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def node_to_dict(self) -> dict:
        def grow_dict(node):
            if node.isleaf():
                return {"label": node.label.tolist()}
            else:
                left = grow_dict(node.left)
                right = grow_dict(node.right)
                return {"left": left, "right": right, "normal": node.normal.tolist()}
    
        return grow_dict(self)
    
    # save tree as json file
    def write(self, file_name: str, debug=False):
        with open(file_name, 'w') as f:
            json.dump(self.node_to_dict(), f, cls=Encoder)
            
        if debug:
            print("Successfully saved trees: {}".format(file_name))            


class ClassificationTree(object):
    def __init__(self, L: int, M: int, k: int, max_in_leaf: int, inverted_index, debug=False):
        self.L = L # label vector space dimension
        self.M = M # feature vector space dimension
        self.k = k # k in `KNNG`
        self.MaxInLeaf = max_in_leaf # max size of leafs in output tree
        self.root = Node() # root node
        self._index = inverted_index # InvertedIndex object
        self._debug = debug

    # make tree from data_set
    def load(self, data_set: dict):
        random_index = random.choice(list(data_set.keys()))
        init_normal = data_set[random_index]["feature"]
        self.root = self._grow_tree(data_set, init_normal)

    # open tree from json file
    def open(self, file_name: str): 
        def grow_node(d):
            if d is None:
                return None
            else:
                left = grow_node(d.get("left"))
                right = grow_node(d.get("right"))
                normal = np.array(d.get("normal")) if d.get("normal") else None
                label = np.array(d.get("label")) if d.get("label") else None

            return Node(left=left, right=right, normal=normal, label=label)
        
        with open(file_name, 'r') as f:
            encoded_dict = json.load(f)
            self.root = grow_node(encoded_dict)  

    def _grow_tree(self, data_set: dict, init_normal: np.ndarray) -> Node:
        if len(data_set) <= self.MaxInLeaf:
            label = self._empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._split_node(data_set, init_normal))

    def _two_valued_classifier(self, sample: np.ndarray, normal: np.ndarray) -> bool:
        return normal @ sample.T > 0

    def _empirical_label_distribution(self, data_set): # -> label vector, np.ndarray object
        if data_set:
            return (1/len(data_set))*np.sum(np.array([data_set[key]["label"] for key in data_set.keys()]), axis=0)
        else:
            return np.zeros(self.L, dtype=int)

    def _split_node(self, data_set: dict, init_normal: np.ndarray) -> tuple:
        L, M, k = self.L, self.M, self.k
        label_vector_dict = {key: data_set[key]["label"] for key in data_set.keys()}
        feature_vector_dict = {key: data_set[key]["feature"] for key in data_set.keys()}

        knn = KNN(L, k, feature_vector_dict)
        lhp = LearnHyperPlane(M, knn, feature_vector_dict, self._index, init_normal, debug=False)

        ### Learning Part ###
        lhp.learn()

        normal = lhp.normal
        left, right = {}, {}
        for key in data_set.keys():
            if self._two_valued_classifier(data_set[key]["feature"], normal):
                left[key] = data_set[key]
            else:
                right[key] = data_set[key]

        left_random_index = random.choice(list(left.keys())) if left else None
        left_init_normal = data_set[left_random_index]["feature"] if left else None
        left_tree = self._grow_tree(left, left_init_normal)

        right_random_index = random.choice(list(right.keys())) if right else None
        right_init_normal = data_set[right_random_index]["feature"] if right else None
        right_tree = self._grow_tree(right, right_init_normal)

        return (left_tree, right_tree, normal)

    def classify(self, sample: np.ndarray): # -> label vector, np.ndarray object
        pointer = self.root
        while not pointer.isleaf():
            normal = pointer.normal
            if self._two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer.label

