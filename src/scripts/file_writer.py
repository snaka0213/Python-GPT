#!/user/bin/env python3
import numpy as np

class FileWriter(object):
    def __init__(self, file_name: str):
        self._file = file_name

    # write on file sorted the top of k-labels
    def write(self, k: int, label_vector_list: list):
        with open(self._file, "w") as f:
            for vector in label_vector_list:
                f.write(",".join([str(index) for index in np.argsort(vector)[::-1][:k]]))
                f.write("\n")

