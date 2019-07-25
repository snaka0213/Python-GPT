#!/user/bin/env python3
import heapq
import numpy as np

from .graph import OrientedGraph

# We assume that `data_set` is a dictionary such that {index: label_vector}
class KNN(object):
    def __init__(self, L: int, k: int, data_set: dict):
        self.L = L
        self.k = k
        self._data_set = data_set
    
    # get list of k-nearest neighbors' index in `index_list`
    # order: <= N*logN
    def get_index(self, query: np.ndarray, index_list: list) -> list:
        k = self.k
        data_set = self._data_set
        
        def sort_key(i) -> np.float64:
            return np.dot(query, data_set[i])/self._label_norm(data_set[i])
        
        knn_list = heapq.nlargest(k, index_list, sort_key)
        return knn_list

    def _label_norm(self, label_vector: np.ndarray):
        return np.sum(label_vector)

