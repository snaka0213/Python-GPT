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
and consider the map f: range(N) -> {0,1}^L and the map Label: {0,1}^L -> P(range(L)).
For given query vector q in {0,1}^L, we define as follows:
** inverted_index of q := f^{-1}(Label^{-1}(Label(q)))
* We decomposite inverted_index with two factors:
** query_labels of q := Label(q)
** self._fiber(*args) := f^{-1}(Label^{-1}(*args))

For given query_index i in range(N), we define
** inverted_index of i := inverted_index of f(i) \ {i}
'''

# TODO: make this order N or L
class InvertedIndex(object):
    def __init__(self, L: int, label_vector_list: list, approximate=False):
        self.L = L
        self.N = len(label_vector_list)
        self._label_vector_list = label_vector_list
        self._approximate = approximate

        N = self.N
        # order: L*N
        self._card_of_fiber_list = [
            np.sum(np.array(
                [label_vector_list[i][l] for i in range(N)], dtype=int
            )) for l in range(L)
        ]
        # order: (L+TH*L*N)*N
        self._index = [self._inverted_index(i) for i in range(N)]

    # returns inverted_index of i
    def get(self, i: int) -> list:
        return self._index[i]

    # order: <= L+TH*L*N
    def _inverted_index(self, query_index: int) -> list:
        L = self.L
        approximate = self._approximate
        card_of_fiber_list = self._card_of_fiber_list
        query = self._label_vector_list[query_index]
        query_labels = [l for l in range(L) if self._hasattr(query, l) \
                        and not(approximate and card_of_fiber_list[l] >= TH)]
        return self._fiber(*query_labels, remove=query_index)

    # order: len(args)*N
    def _fiber(self, *args, remove=None) -> list:
        N = self.N
        label_vector_list = self._label_vector_list
        return [i for i in range(N) if self._hasattr(label_vector_list[i], *args) and i != remove]

    # order: <= len(args)
    def _hasattr(self, label_vector, *args) -> bool:
        for index in args:
            if label_vector[index] > 0:
                return True
        else:
            return False

        
class KNN(object):
    def __init__(self, L: int, label_vector_list: list):
        self.L = L
        self.N = len(label_vector_list)
        self._label_vector_list = label_vector_list

    '''
    Here we assume that k << N
    * use `heapq` module, order N+k*logN
    * it is faster than normal sort, order N*logN
    '''
    
    def knn_index(self, k: int, query_index: int, inverted_index) -> list:
        N = self.N
        label_vector_list = self._label_vector_list
        query = label_vector_list[query_index]
        
        def sort_key(i: int) -> np.float64:
            return np.dot(query, label_vector_list[i])/self._label_norm(label_vector_list[i])
        
        knn_list = heapq.nlargest(k, inverted_index, key=sort_key)
        return knn_list

    def _label_norm(self, label_vector: np.ndarray):
        return np.sum(label_vector)


class KNNG(object):
    def __init__(self, k: int, L: int, label_vector_list: list):
        self.k = k
        self.L = L
        self.N = len(label_vector_list)
        self._label_vector_list = label_vector_list

    def get_graph(self, approximate=False) -> OrientedGraph:
        k, L, N = self.k, self.L, self.N
        label_vector_list = self._label_vector_list

        index = InvertedIndex(L, label_vector_list, approximate)
        oriented_graph = OrientedGraph(N)
        for i in range(N):
            knn = KNN(L, label_vector_list)
            knn_index = knn.knn_index(k, i, index.get(i))
            for j in knn_index:
                oriented_graph.add_edge(i, j)

        return oriented_graph
