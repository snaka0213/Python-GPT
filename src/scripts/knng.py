#!/user/bin/env python3
import settings
import numpy as np
from .graph import OrientedGraph

TH = settings.ThresholdParameter

class KNN(object):
    def __init__(self, L: int, labels: list, approximate=False):
        self.L = L
        self._labels = labels
        self._approximate = approximate

    # if approximate, order < L*TH
    def knn_index(self, k: int, query: np.ndarray) -> list:
        labels = self._labels
        inverted_index = self._inverted_index(query)
        tmp_list = sorted([
            {
                "index": i,
                "value": self._num_of_intersection(query, labels[i])/self._label_norm(labels[i])
            } for i in inverted_index
        ], key=lambda e:e["value"], reverse=True)
        return map(lambda e:e["index"], tmp_list[:k])

    def _inverted_index(self, query: np.ndarray) -> list:
        L = self.L
        labels = self._labels
        approximate = self._approximate
        
        query_labels = [l for l in range(L) if self._hasattr(query, l) and not(approximate and len(self._fiber(l)) >= TH)]
        return self._fiber(*query_labels)

    def _fiber(self, *args) -> list:
        labels = self._labels
        return [i for i in range(len(labels)) if self._hasattr(labels[i], *args)]

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
    def __init__(self, k: int, L: int, labels: list, approximate=False):
        self.k = k
        self.L = L
        self._labels = labels
        self._approximate = approximate

    def graph(self) -> OrientedGraph:
        k = self.k
        L = self.L
        labels = self._labels
        approximate = self._approximate
        
        oriented_graph = OrientedGraph(len(labels))
        for i in range(len(labels)):
            knn = KNN(L, labels[:i]+labels[i+1:], approximate) if i != 0 else KNN(L, labels[1:], approximate)
            knn_index = knn.knn_index(k, labels[i])
            for j in knn_index:
                if j < i:
                    oriented_graph.add_edge(i, j)
                else:
                    oriented_graph.add_edge(i, j+1)

        return oriented_graph


