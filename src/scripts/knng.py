#!/user/bin/env python3
import settings
import numpy as np
from .graph import OrientedGraph

TH = settings.ThresholdParameter

class KNN(object):
    def __init__(self, L: int, label_vector_list: list, approximate=False):
        self.L = L
        self._label_vector_list = label_vector_list
        self._approximate = approximate

    # if approximate, order < L*TH
    def knn_index(self, k: int, query: np.ndarray) -> list:
        label_vector_list = self._label_vector_list
        inverted_index = self._inverted_index(query)
        tmp_list = sorted([
            {
                "index": i,
                "value": self._num_of_intersection(query, label_vector_list[i])/self._label_norm(label_vector_list[i])
            } for i in inverted_index
        ], key=lambda e:e["value"], reverse=True)
        return map(lambda e:e["index"], tmp_list[:k])

    def _inverted_index(self, query: np.ndarray) -> list:
        L = self.L
        label_vector_list = self._label_vector_list
        approximate = self._approximate

        query_label_vector_list = [l for l in range(L) if self._hasattr(query, l) and not(approximate and len(self._fiber(l)) >= TH)]
        return self._fiber(*query_label_vector_list)

    def _fiber(self, *args) -> list:
        label_vector_list = self._label_vector_list
        return [i for i in range(len(label_vector_list)) if self._hasattr(label_vector_list[i], *args)]

    def _num_of_intersection(self, v: np.ndarray, w: np.ndarray) -> int:
        num = 0
        L = self.L
        for i in range(L):
            if self._hasattr(v, i) and self._hasattr(w, i):
                num += 1

        return num

    def _hasattr(self, label, *args) -> bool:
        for index in args:
            if label[index] > 0:
                return True
        else:
            return False

    def _label_norm(self, label: np.ndarray):
        return np.sum(label)


class KNNG(object):
    def __init__(self, k: int, L: int, label_vector_list: list, approximate=False):
        self.k = k
        self.L = L
        self._label_vector_list = label_vector_list
        self._approximate = approximate

    def graph(self) -> OrientedGraph:
        k = self.k
        L = self.L
        label_vector_list = self._label_vector_list
        approximate = self._approximate

        oriented_graph = OrientedGraph(len(label_vector_list))
        for i in range(len(label_vector_list)):
            knn = KNN(L, label_vector_list[:i]+label_vector_list[i+1:], approximate) if i != 0 else KNN(L, label_vector_list[1:], approximate)
            knn_index = knn.knn_index(k, label_vector_list[i])
            for j in knn_index:
                if j < i:
                    oriented_graph.add_edge(i, j)
                else:
                    oriented_graph.add_edge(i, j+1)

        return oriented_graph
