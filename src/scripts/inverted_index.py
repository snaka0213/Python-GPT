#!/user/bin/env python3
import numpy as np

'''
__Assume__
* Given a dictionary `data_set` has data structure such as:
    {index: data}, and each `data` has attribute 'label':
    data['label']: list object, subset in [0,...,L-1]

__Terminology__
* L: int object called `label space dimension`
* label: list of int object in range(L) := {0,...,L-1}

* inverted_index: Let N := len(data_set).
* For a given label l in range(L), we define Inv[l] a subset of range(N):
    Inv[l] := {i for in range(N) if l in data_set[i]['label']}.
'''

class InvertedIndex(object):
    def __init__(self, *, L: int):
        self.L = L # label space dimension
        self._inverted_index = [] # [Inv[l] for l in range(L)]

    def _hasattr(self, label, l) -> bool:
        return l in label

    # order: L*N
    def make_index(self, data_set: dict):
        L = self.L
        self._inverted_index = [self._fiber(data_set, l) for l in range(L)]

    # returns Inv[l]
    def get(self, l) -> list:
        return self._inverted_index[l]

    # order: N
    def _fiber(self, data_set: dict, l: int) -> list:
        fiber = [key for key in data_set.keys() if self._hasattr(data_set[key]['label'], l)]
        return fiber

    ### File Writer ###
    # save inverted_index as a new .txt file
    # 1 line per label in range(L): idx1,idx2,...,idxk
    def write(self, file_name: str):
        L = self.L
        with open(file_name, 'w') as f:
            for l in range(L):
                line = ','.join(str(i) for i in self.get(l))
                f.write(line+'\n')

        print("Successfully saved inverted index file: {}".format(file_name))

    ### File Reader ###
    # save inverted_index in self from a .txt file
    def read(self, file_name: str):
        L = self.L
        with open(file_name, 'r') as f:
            line = f.readline()
            while line:
                self._inverted_index.append([int(x) for x in line.split(',')])
                line = f.readline()

        print("Successfully loaded inverted index file: {}".format(file_name))
