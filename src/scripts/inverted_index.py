#!/user/bin/env python3
import json
import heapq
import numpy as np

import settings
TH = settings.ThresholdParameter

'''
Let L be an int object, called as `label space dimension`.
__Terminology__
* label: int object in range(L)
* labels: list object, consist of some labels
* label_vector: np.ndarray object, size L

* inverted_index: Let N := len(data_set)
and consider the map f: range(N) -> P(range(L)) which maps i to labels of data_set[i].
For given label l in range(L), we define fiber[l] a subset of range(N) as
** fiber[l] := [i for in range(N) if l in f(i)]
For given query vector q in {0,1}^L=P(range(L)), we define as follows:
** inverted_index[q] := sum_{l in L} X[l]
For given query_index i in range(N), we define
** inverted_index[i] := inverted_index[f(i)] \ {i}

__Assume__
* Given a dictionary `data_set` has structure such that {index: data}:
** data["label"]: label_vector, np.ndarray object, size L
'''

class InvertedIndex(object):
    def __init__(self, L: int, data_set: dict, approximate: bool = False):
        self.L = L
        self._data_set = data_set
        self._approximate = approximate
        self.index_dict = None
    
        # order: L*N
        self._fiber_list = [self._fiber(l) for l in range(L)]
        
        # order: <= L*N, max length of elements: <= TH*L
        self.index_dict = {key: self._inverted_index(key) for key in data_set.keys()}

    # returns inverted_index of key
    def get(self, key) -> list:
        return self.index_dict[key]

    # order: len(args)*N
    def _fiber(self, *args, remove=None) -> list:
        data_set = self._data_set
        
        s = set()
        for label in args:
            s = s | {key for key in data_set.keys() if self._hasattr(data_set[key]["label"], label)}
        if remove in s:
            s.remove(remove)
        return list(s)

    # order: <= L*N
    def _inverted_index(self, query_index: int) -> list:
        L = self.L
        approximate = self._approximate
        fiber_list = self._fiber_list
        query = self._data_set[query_index]["label"]
        query_labels = [l for l in range(L) if self._hasattr(query, l) \
                        and not(approximate and len(fiber_list[l]) >= TH)]
        return self._fiber(*query_labels, remove=query_index)

    # order: 1
    def _hasattr(self, label_vector, label) -> bool:
        return label_vector[label] > 0

    ### File Writer ###
    # save inverted_index as a new json file, list of
    # {index: inverted_index[index] (: list object)}
    def write(self, file_name: str):
        with open(file_name, 'w') as f:
            json.dump(self.index_dict, f, ensure_ascii=False, indent=4, separators=(',', ': '))
    
    ### File Reader ###
    # save inverted_index in self from a json file
    def open(self, file_name: str):
        with open(file_name, 'r') as f:
            self.index_dict = json.load(f)

    
