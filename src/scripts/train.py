#!/usr/bin/env python3
import numpy as np
from .learning.classifier import ClassificationTree
from .inverted_index import InvertedIndex
from .file_reader import FileReader
from .file_writer import FileWriter

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

    def make_index(self, TH=False, debug=False) -> InvertedIndex:
        inverted_index = InvertedIndex(self.L, self._data_set, TH)
        if debug:
            print("Inverted index has been made.")
        return inverted_index
    
    def make_tree(self, T: int, k: int, max_in_leaf: int, inverted_index, debug=False): # -> T-tuple of Trees
        data_set, L, M = self._data_set, self.L, self.M
        tree_list = [ClassificationTree(L, M, k, max_in_leaf, inverted_index, debug) for i in range(T)]
        for i in range(T):
            tree_list[i].load(data_set)
            if debug:
                print("Tree{} Learning: Done.".format(i))

        return tuple(tree_list)
