#!/usr/bin/env python3

class Validate(object):
    def __init__(self, k: int, N: int):
        self.k = k
        self.N = N
        self.predict_list = []

    # each element in `self.predict_list` is the form: label_lth (l=1, ..., k)
    def read(self, file_name: str):
        with open(file_name, "r") as f:
            line = f.readline()
            while line:
                self.predict_list.append([int(x) for x in line.split(",")])
                line = f.readline()

    def diff(self, label_vector_list: list):
        k, N, predict_list = self.k, self.N, self.predict_list
        
        hit_counter = 0
        for i in range(N):
            label_vector = label_vector_list[i]
            predict_labels = predict_list[i]

            for label in predict_labels:
                hit_counter += label_vector[label]

        print("Correct Rate: {}".format(hit_counter/(k*N)))
