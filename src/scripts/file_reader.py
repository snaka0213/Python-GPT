#!/user/bin/env python3
import numpy as np

'''
make `data_set`, dict object {index: data}
* `data` is dict object, whose keys are:
**  "label" : label_vector, size=L
** "feature": feature_vector, size=M
'''

class FileReader(object):
    def __init__(self, file_name: str):
        self._file = file_name
        self.L = None # label space dimension
        self.M = None # feature vector space dimension
        self.N = None # the size of data_set

    def read(self) -> dict:
        with open(self._file, "r") as f:
            header = f.readline()
            N, M, L = (int(x) for x in header.split())
            self.L, self.M, self.N = L, M, N

            line = f.readline()
            data_set, index = {i:{} for i in range(N)}, 0

            while line:
                data = {
                     "label" : np.zeros(L, dtype=int),
                    "feature": np.zeros(M),
                }
                zero_term = line.split()[0]
                if not ':' in zero_term:
                    start_index = 1
                    labels = [int(x) for x in zero_term.split(",")]
                else:
                    start_index = 0
                    labels = []

                for label in labels:
                    data["label"][label] = 1
                    
                for x in line.split()[start_index:]:
                    data["feature"][int(x.split(":")[0])] = float(x.split(":")[1])

                data_set[index] = data  
                line = f.readline()
                index += 1

        return data_set
