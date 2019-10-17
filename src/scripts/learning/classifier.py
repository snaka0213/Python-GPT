#!/user/bin/env python3
import json
import random
import numpy as np

from .knng import KNNG
from .learn_hyperplane import LearnHyperPlane

'''
__Assume__
* Given `data_set`, dict object {index: data}
    `data` is dict object, whose keys are:
    *  'label' : `label_vector`, list object subset in [0,...,L-1]
    * 'feature': `feature_vector`, dict object {coordinate index: value}
'''

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
        self.left = left # left subtree
        self.right = right # right subtree
        self.normal = normal # positive normal vector of hyperplane, dict object
        self.label = label # weighted label vector, np.ndarray object

    def isleaf(self) -> bool:
        return self.left is None and self.right is None

    def node_to_dict(self) -> dict:
        return self._grow_dict(self)

    def _grow_dict(self, node):
        if node.isleaf():
            return {'label': node.label.tolist()}
        else:
            left = self._grow_dict(node.left)
            right = self._grow_dict(node.right)
            return {'left': left, 'right': right, 'normal': node.normal}


class ClassificationTree(object):
    def __init__(self, *, L: int, M: int, k: int, max_in_leaf: int):
        self.L = L # label vector space dimension
        self.M = M # feature vector space dimension
        self.k = k # the number of each kNN
        self.MaxInLeaf = max_in_leaf # max size of leafs in output tree
        self.root = Node() # root node

    # make tree from data_set
    def make_tree(self, data_set: dict, inverted_index):
        random_index = random.choice(list(data_set.keys()))
        init_normal = data_set[random_index]['feature']
        self.root = self._grow_tree(data_set, inverted_index)

    def _grow_tree(self, data_set: dict, inverted_index) -> Node:
        if len(data_set) <= self.MaxInLeaf:
            label = self._empirical_label_distribution(data_set)
            return Node(label=label)
        else:
            return Node(*self._split_node(data_set, inverted_index))

    def _split_node(self, data_set: dict, inverted_index) -> tuple:
        G = KNNG(k=self.k, L=self.L, data_set=data_set, inverted_index=inverted_index)
        H = LearnHyperPlane(M=self.M, G=G.get_graph(), data_set=data_set)
        H.learn(debug=False)

        left, right = {}, {}
        for key in data_set.keys():
            if self._two_valued_classifier(data_set[key]['feature'], H.normal):
                left[key] = data_set[key]
            else:
                right[key] = data_set[key]

        left_tree = self._grow_tree(left, inverted_index)
        right_tree = self._grow_tree(right, inverted_index)

        return (left_tree, right_tree, H.normal)

    def classify(self, sample: dict) -> np.ndarray:
        pointer = self.root
        while not pointer.isleaf():
            normal = pointer.normal
            if self._two_valued_classifier(sample, normal):
                pointer = pointer.left
            else:
                pointer = pointer.right

        else:
            return pointer.label

    def _two_valued_classifier(self, v: dict, w: dict) -> bool:
        value = 0
        for key in v.keys() & w.keys():
            value += v[key]*w[key]

        return value > 0

    def _empirical_label_distribution(self, data_set) -> np.ndarray:
        label_vector = np.zeros(self.L)
        n = len(data_set)
        for key in data_set.keys():
            for l in data_set[key]['label']:
                label_vector[l] += 1/n

        return label_vector

    ### File Writer ###
    # save ClassificationTree as a new json file
    def write(self, file_name: str):
        with open(file_name, 'w') as f:
            json.dump(self.root.node_to_dict(), f, cls=Encoder)

        print("Successfully saved tree: {}".format(file_name))

    ### File Reader ###
    # save ClassificationTree in self from a json file
    # Notice: key in json is ALWAYS strings
    def read(self, file_name: str):
        with open(file_name, 'r') as f:
            encoded_dict = json.load(f)
            self.root = self._dict_to_node(encoded_dict)

        print("Successfully loaded tree: {}".format(file_name))

    def _dict_to_node(self, d: dict) -> Node:
        if d is None:
            return None
        else:
            left = self._dict_to_node(d.get('left'))
            right = self._dict_to_node(d.get('right'))
            if d.get('normal') is None:
                normal = None
            else:
                normal = {}
                for key in d['normal'].keys():
                    normal[int(key)] = d['normal'][key]

            label = np.array(d.get('label')) if d.get('label') is not None else None
            return Node(left=left, right=right, normal=normal, label=label)
