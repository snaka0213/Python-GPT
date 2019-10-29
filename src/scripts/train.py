#!/usr/bin/env python3
import numpy as np
from .learning.classifier import ClassificationTree
from .inverted_index import InvertedIndex
from .file_reader import FileReader

class Train(object):
    def __init__(self):
        self.L = None
        self.M = None
        self.N = None
        self._data_set = {}

    def read(self, file_name: str):
        reader = FileReader()
        self._data_set = reader.read(file_name)
        self.L, self.M, self.N = reader.L, reader.M, reader.N
        print("Train data has been loaded.")

    def make_index(self) -> InvertedIndex:
        inverted_index = InvertedIndex(L=self.L)
        inverted_index.make_index(self._data_set)
        print("Inverted index has been made.")
        return inverted_index

    def make_tree(self, *, k: int, TH: int, max_in_leaf: int, inverted_index: InvertedIndex) -> ClassificationTree:
        tree = ClassificationTree(L=self.L, M=self.M, k=k, TH=TH, max_in_leaf=max_in_leaf)
        tree.make_tree(data_set=self._data_set, inverted_index=inverted_index)
        return tree
