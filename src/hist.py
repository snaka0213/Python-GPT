#!/usr/bin/env python3
import os
import sys
import matplotlib.pyplot as plt

from scripts.train import Train
from scripts.inverted_index import InvertedIndex

if __name__ == '__main__':
    path = input("Data set file name: ")
    train_file = "data/" + path + "/train.txt"

    # train_data read
    train = Train()
    train.read(train_file)

    inverted_index = InvertedIndex(L=train.L, data_set=train._data_set, TH=-1)
    inverted_index.make_fiber()
    x = [len(fiber) for fiber in inverted_index._fiber_list]
    plt.hist(x, bins=max(max(x)//10, 10), range=(0, max(x)))
    plt.show()
