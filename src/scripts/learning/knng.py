#!/user/bin/env python3
import copy
import heapq
import random
import numpy as np

from .graph import OrientedGraph

'''
__Assume__
* Given `data_set`, dict object {index: data}
    `data` is dict object, whose keys are:
    *  'label' : `label_vector`, list object subset in [0,...,L-1]
    * 'feature': `feature_vector`, dict object {coordinate index: value}
'''

class KNNG(object):
    def __init__(self, *, k: int, L: int, data_set: dict, inverted_index):
        self.k = k
        self.L = L
        self._data_set = data_set
        self._inverted_index = inverted_index

    def get_graph(self) -> OrientedGraph:
        k = self.k
        data_set = self._data_set
        inverted_index = self._inverted_index

        G = OrientedGraph(nodes=list(data_set.keys()))
        for i in G.nodes:
            candidate_list = [j for j in inverted_index.get(i) if j in data_set.keys()]
            N = heapq.nlargest(k, candidate_list, self._sort_key(i))

            # make to be `len(N) = k` version
            if len(N) < k:
                temp = copy.copy(list(data_set.keys()))
                for x in N:
                    temp.remove(x)
                else:
                    complement = random.sample(temp, k-len(N))
                    N.extend(complement)

            for j in N:
                G.add_edge(j, i)

        return G

    def _sort_key(self, i: int):
        L = self.L

        def _func(j: int) -> float:
            label_i = self._data_set[i]['label']
            label_j = self._data_set[j]['label']
            cap = set(label_i) & set(label_j)
            return len(cap)/(len(label_i)*len(label_j))

        return _func
