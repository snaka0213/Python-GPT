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
        self._data_set = None

    def read(self, file_name: str):
        reader = FileReader()
        self._data_set = reader.read(file_name)
        self.L, self.M, self.N = reader.L, reader.M, reader.N
        print("Train data has been loaded.")

    def make_index(self, TH: int) -> InvertedIndex:
        inverted_index = InvertedIndex(L=self.L, data_set=self._data_set, TH=TH)
        inverted_index.make_index()
        print("Inverted index has been made.")
        return inverted_index

    def make_tree(self, *, k: int, max_in_leaf: int, inverted_index: InvertedIndex) -> ClassificationTree:
        tree = ClassificationTree(L=self.L, M=self.M, k=k, max_in_leaf=max_in_leaf)
        tree.make_tree(self._data_set, inverted_index)
        return tree
