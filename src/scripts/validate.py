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

        for i in range(1, k+1):
            hit_counter = 0
            for j in range(N):
                label_vector = label_vector_list[j]
                predict_labels = predict_list[j]

                for label in predict_labels[:i]:
                    hit_counter += label_vector[label]

            print("Precision@{} Correct Rate: {}".format(i, hit_counter/(i*N)))
