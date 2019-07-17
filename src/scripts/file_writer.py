#!/user/bin/env python3
import numpy as np

class FileWriter(object):
    def __init__(self, file_name: str):
        self._file = file_name

    '''
    write on file sorted k-indexes of labels
    * assume that each elements in `label_list` is np.ndarray
    '''
    def write(self, k: int, label_vectors_list: list):
        with open(self._file, "w") as f:
            for vector in label_vectors_list:
                f.write(",".join([str(index) for index in np.argsort(vector)[::-1][:k]]))
                f.write("\n")

