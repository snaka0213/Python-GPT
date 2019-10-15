#!/usr/bin/env python3
class Validate(object):
    def __init__(self, k: int, N: int):
        self.k = k
        self.N = N
        self.prediction_list = []

    # each element in `self.prediction_list` is the form: label_lth (l=1, ..., k)
    def read(self, file_name: str):
        with open(file_name, "r") as f:
            line = f.readline()
            while line:
                self.prediction_list.append([int(x) for x in line.split(",")])
                line = f.readline()

    def diff(self, label_list: list):
        k, N, prediction_list = self.k, self.N, self.prediction_list

        for i in range(1, k+1):
            hit_counter = 0
            for j in range(N):
                label = label_list[j]
                prediction = prediction_list[j]

                for l in prediction[:i]:
                    hit_counter += 1 if l in label else 0

            print("Precision@{} Correct Rate: {}".format(i, hit_counter/(i*N)))
