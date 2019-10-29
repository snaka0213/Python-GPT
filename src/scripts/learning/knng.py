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

* TH := ThresholdParameter: int object which make
    len(Inv[l]) < TH for each l in range(L)
* If TH = -1, then make KNNG by brute force.
'''

class KNNG(object):
    def __init__(self, *, L: int, k: int, TH: int, data_set: dict, inverted_index):
        self.L = L
        self.k = k
        self.TH = TH
        self._data_set = data_set
        self._inverted_index = inverted_index

    def is_approximate(self):
        return self.TH != -1

    def get_graph(self) -> OrientedGraph:
        L = self.L
        k = self.k
        TH = self.TH
        data_set = self._data_set
        inverted_index = self._inverted_index

        G = OrientedGraph(nodes=list(data_set.keys()))
        for i in G.nodes:
            score_dict = {}
            for l in data_set[i]['label']:
                Inv_l = inverted_index.get(l) & data_set.keys() # set object
                if self.is_approximate() and len(Inv_l) >= TH:
                    pass
                else:
                    for j in Inv_l:
                        label_j = data_set[j]['label']
                        if j == i:
                            pass
                        else:
                            if j in score_dict.keys():
                                score_dict[j] += 1/len(label_j)
                            else:
                                score_dict[j] = 1/len(label_j)

            N = heapq.nlargest(k, score_dict.keys(), key=lambda j:score_dict[j])

            # make to be `len(N) = k` version
            if len(N) < k:
                temp = copy.copy(list(data_set.keys()))
                temp.remove(i)
                for x in N:
                    temp.remove(x)
                complement = random.sample(temp, k-len(N))
                N.extend(complement)

            for j in N:
                G.add_edge(i, j)

        return G
