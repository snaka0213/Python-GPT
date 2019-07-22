#!/user/bin/env python3
import heapq
import numpy as np
import settings
from .graph import OrientedGraph

TH = settings.ThresholdParameter

'''
__Terminology__
* label: int object in range(L)
* labels: list object, element is label
* label_vector: np.ndarray object, size is L

* inverted_index: Let N := len(data_set)
and consider the map f: range(N) -> P(range(L)) which maps i to labels of data_set[i].
For given label l in range(L), we define fiber[l] a subset of range(N) as
** fiber[l] := [i for in range(N) if l in f(i)]
For given query vector q in {0,1}^L=P(range(L)), we define as follows:
** inverted_index[q] := sum_{l in L} X[l]
For given query_index i in range(N), we define
** inverted_index[i] := inverted_index[f(i)] \ {i}
'''

class InvertedIndex(object):
    def __init__(self, L: int, data_set: dict, approximate=False):
        self.L = L
        self.N = len(data_set)
        self._data_set = data_set
        self._approximate = approximate
        
        # order: L*N
        self._fiber_list = [self._fiber(l) for l in range(L)]
        
        # order: <= L*N, max length of elements: <= TH*L
        N = self.N
        self._index = [self._inverted_index(i) for i in range(N)]

    # returns inverted_index of i
    def get(self, i: int) -> set:
        return self._index[i]

    # order: len(args)*N
    def _fiber(self, *args, remove=None) -> set:
        N = self.N
        data_set = self._data_set
        
        s = set()
        for label in args:
            s = s | {key for key in data_set if self._hasattr(data_set[key]["label"], label)}
        if remove in s:
            s.remove(remove)
        return s

    # order: <= L*N
    def _inverted_index(self, query_index: int) -> set:
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

    
class KNN(object):
    def __init__(self, L: int, data_set: dict):
        self.L = L
        self.N = len(data_set)
        self._data_set = data_set
    
    def knn_index(self, k: int, query: np.ndarray, index_list: list) -> list:
        N = self.N
        data_set = self._data_set
        
        def sort_key(i: int) -> np.float64:
            return np.dot(query, data_set[i]["label"])/self._label_norm(data_set[i]["label"])
        
        knn_list = heapq.nlargest(k, index_list, sort_key)
        return knn_list

    def _label_norm(self, label_vector: np.ndarray):
        return np.sum(label_vector)


class KNNG(object):
    def __init__(self, k: int, L: int, data_set: list, inverted_index: InvertedIndex):
        self.k = k
        self.L = L
        self.N = len(data_set)
        self._data_set = data_set
        self._index = inverted_index

    def get_graph(self) -> OrientedGraph:
        k, L, N = self.k, self.L, self.N
        data_set = self._data_set
        index = self._index

        index_in_data_set = {key for key in data_set}
        oriented_graph = OrientedGraph([key for key in data_set])
        for i in data_set:
            knn = KNN(L, data_set)
            index_list = list(self._index.get(i)&index_in_data_set)
            knn_index = knn.knn_index(k, data_set[i]["label"], index_list)
            for j in knn_index:
                oriented_graph.add_edge(i, j)

        return oriented_graph
