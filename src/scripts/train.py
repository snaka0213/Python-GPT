#!/usr/bin/env python3
import numpy as np
#from .learning.classifier import ClassificationTree
from .file_reader import FileReader

class Train(object):
    def __init__(self):
        self.L = None
        self.M = None
        self._data_set = None

    def load(self, file_name: str, debug=False):
        reader = FileReader(file_name)
        self._data_set, self.L, self.M = reader.read(), reader.L, reader.M
        if debug:
            print("Train Data has been loaded.")

    def make_tree(self, T: int, debug=False): # -> T-tuple of Trees
        data_set, L, M = self._data_set, self.L, self.M
        inverted_index = InvertedIndex(L, data_set, approximate=True)
        if debug:
            print("InvertedIndex has been made.")
        tree_list = [ClassificationTree(L, M, inverted_index) for i in range(T)]
        for i in range(T):
            tree_list[i].load(data_set)
            if debug:
                print("Tree{} Learning: Done.".format(i))

        return tuple(tree_list)
