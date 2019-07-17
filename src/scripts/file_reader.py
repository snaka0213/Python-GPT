#!/user/bin/env python3
import numpy as np

class FileReader(object):
    def __init__(self, file_name: str):
        self._file = file_name
        self.L = None # label space dimension
        self.M = None # feature vector space dimension
        self.N = None # the number of data

    def read(self) -> list:
        data_set = []
        with open(self._file, "r") as f:
            header = f.readline()
            N, M, L = (int(x) for x in header.split())
            self.L, self.M, self.N = L, M, N

            line = f.readline()
            while line:
                data = {
                    "label": np.zeros(L, dtype=int),
                    "feature": np.zeros(M)
                }
                labels = [int(x) for x in line.split()[0].split(",")]
                for index in labels:
                    data["label"][index] += 1
                    
                for x in line.split()[1:]:
                    data["feature"][int(x.split(":")[0])] = float(x.split(":")[1])
                    
                data_set.append(data)
                line = f.readline()
            
        return data_set
