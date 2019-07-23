#!/user/bin/env python3
from .graph import OrientedGraph

class KNN(object):
    def __init__(self, L: int, data_set: dict):
        self.L = L
        self._data_set = data_set
    
    def knn_index(self, k: int, query: np.ndarray, index_list: list) -> list:
        data_set = self._data_set
        
        def sort_key(i) -> np.float64:
            return np.dot(query, data_set[i]["label"])/self._label_norm(data_set[i]["label"])
        
        knn_list = heapq.nlargest(k, index_list, sort_key)
        return knn_list

    def _label_norm(self, label_vector: np.ndarray):
        return np.sum(label_vector)


class KNNG(object):
    def __init__(self, k: int, L: int, data_set: dict,  inverted_index: InvertedIndex):
        self.k = k
        self.L = L
        self._data_set = data_set
        self._inverted_index = inverted_index

    def get_graph(self) -> OrientedGraph:
        k, L = self.k, self.L
        data_set = self._data_set
        inverted_index = self._inverted_index

        index_in_data_set = set(data_set.keys())
        oriented_graph = OrientedGraph([key for key in data_set.keys()])
        for i in data_set.keys():
            knn = KNN(L, data_set)
            index_list = list(inverted_index.get(i)&index_in_data_set)
            knn_index = knn.knn_index(k, data_set[i]["label"], index_list)
            for j in knn_index:
                oriented_graph.add_edge(i, j)

        return oriented_graph