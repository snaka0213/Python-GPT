#!/user/bin/env python3
import json
import heapq
import numpy as np

'''
__Assume__
* Given a dictionary `data_set` has data structure such as:
    {index: data}, and each `data` has attribute 'label':
    data['label']: list object, subset in [0,...,L-1]

__Terminology__
* L: int object called `label space dimension`
* label: list of int object in range(L) := {0,...,L-1}

* inverted_index: Let N := len(data_set) and consider the map
    f: range(N) -> P(range(L)), i -> data_set[i]['label'].

    * For a given label l in range(L), we define fiber[l],
        a subset of range(N), as fiber[l] := {i for in range(N) if l in f(i)}.
    * For given query vector q in P(range(L)), we define as follows:
        inverted_index[q] := sum_{l in q} fiber[l].
    * For given query_index i in range(N), we define
        inverted_index[i] := inverted_index[f(i)] \ {i}.

* TH := ThresholdParameter: int object which make
    len(fiber[l]) < TH for each l in range(L)
* If TH = -1, then make inverted index by brute force.

'''

class InvertedIndex(object):
    def __init__(self, *, L: int, data_set: dict, TH: int):
        self.L = L # label space dimension
        self.TH = TH # the ThresholdParameter
        self._data_set = data_set # the original data_set
        self._fiber_list = [] # [fiber[l] for l in range(L)]
        self._index_dict = {} # {index: inverted_index[index]}

    def is_approximate(self) -> bool:
        return self.TH != -1

    def _hasattr(self, label, l) -> bool:
        return l in label

    # order: L*N
    def make_index(self):
        L = self.L
        data_set = self._data_set
        self._fiber_list = [self._fiber(l) for l in range(L)]

        for key in data_set.keys():
            self._index_dict[key] = self._inverted_index(key)

    # returns inverted_index[key]
    def get(self, key) -> list:
        return self._index_dict[key]

    # order: N
    def _fiber(self, l) -> set:
        data_set = self._data_set
        fiber = {key for key in data_set.keys() if self._hasattr(data_set[key]['label'], l)}
        return fiber

    def _inverted_index(self, query_index: int) -> list:
        TH = self.TH
        fiber_list = self._fiber_list
        original_label = self._data_set[query_index]['label']

        index_set = set()
        for l in original_label:
            if not(self.is_approximate() and len(fiber_list[l]) >= TH):
                index_set = index_set | fiber_list[l]

        index_set.discard(query_index)
        return list(index_set)

    ### File Writer ###
    # save inverted_index as a new json file, list of
    # {index: inverted_index[index] (: list object)}
    def write(self, file_name: str):
        with open(file_name, 'w') as f:
            json.dump(self._index_dict, f)

        print("Successfully saved inverted index file: {}".format(file_name))

    ### File Reader ###
    # save inverted_index in self from a json file
    # Notice: key in json is ALWAYS strings
    def read(self, file_name: str):
        with open(file_name, 'r') as f:
            encoded_dict = json.load(f)
            for key in encoded_dict.keys():
                self._index_dict[int(key)] = encoded_dict[key]

        print("Successfully loaded inverted index file: {}".format(file_name))
