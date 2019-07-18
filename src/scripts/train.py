#!/usr/bin/env python3
import numpy as np
from .classifier import ClassificationTree
from .file_reader import FileReader

class Train(object):
    def __init__(self):
        self.data_set = None
        self.L = None
        self.M = None

    def load(self, file_name: str,  debug=False):
        reader = FileReader(file_name)
        self.data_set, self.L, self.M = reader.read(), reader.L, reader.M
        if debug:
            print("Train Data has been loaded.")

    def make_tree(self, T: int, debug=False): # -> T-tuple of Trees
        L, M = self.L, self.M
        tree_list = [ClassificationTree(L, M) for i in range(T)]
        for i in range(T):
            tree_list[i].load(self.data_set)
            if debug:
                print("Tree{} Learning: Done.".format(i))

        return tuple(tree_list)
