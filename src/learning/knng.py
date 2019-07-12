#!/user/bin/env python3
import settings
import numpy as np
from ..graph.graph import OrientedGraph

L = settings.LabelSpaceDimension
ThresholdParameter = settings.ThresholdParameter

class KNN(object):
    def __init__(self):
        self._labels = []

    def load(self, labels: list):
        self._labels = labels

    ''' order L*N*logN
    def knn_index(self, k: int, label: np.ndarray) -> list:
        _labels = self._labels
        _list = [
            {
                "index": i,
                "value": self._label_product(label, _labels[i])
            } for i in range(len(_labels))
        ].sorted(key=lambda e:e["value"])
        return map(lambda e:e["index"], _list[:k])
    '''

    def knn_index(self, k: int, label: np.ndarray) -> list:
        _labels = self._labels
        for j in range(L):
            for _data in self._inverted_index_data_set(j):
                # TODO: make here
                pass
                

    def _inverted_index_data_set(self, j: int) -> list:
        def _hasattr(label, index):
            return label[index] > 0

        _labels = self._labels
        return [_label for _label in _labels if _hasattr(_label, j)]

    def _label_product(self, v: np.ndarray, w: np.ndarray):
        return np.dot(v, w)/(np.sum(v+w,axis=0))

        
class KNNG(object):
    def __init__(self):
        pass
    
    def graph(self, k: int, data_set: list) -> OrientedGraph:
        graph = OrientedGraph(len(data_set))
        
        _knn = KNN()
        for i in range(len(data_set)):
            _knn.load(data_set[:i]+data_set[i+1:]) if i != 0 else _knn.load(data_set[1:])
            _knn_index = _knn.knn_index(k, data_set[i])
            for j in _knn_index:
                if j < i:
                    graph.add_edge(i, j)
                else:
                    graph.add_edge(i, j+1)

        return graph
            
class ApproximateKNNG(object):
    def __init__(self):
        pass
