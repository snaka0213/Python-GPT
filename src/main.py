#!/usr/bin/env python3
import os
import sys
import numpy as np
from multiprocessing import Pool

import settings
from scripts.train import Train
from scripts.predict import Predict
from scripts.validate import Validate
from scripts.file_reader import FileReader
from scripts.inverted_index import InvertedIndex
from scripts.learning.classifier import ClassificationTree

if __name__ == '__main__':
    path = input("Data set file name: ")
    train_file = "data/" + path + "/train.txt"
    index_file = "data/" + path + "/index.txt"
    predict_file = "data/" + path + "/test.txt"
    trees_dir = "data/" + path + "/trees/"

    def tree_file_name(i: int):
        return "{}tree_{}.json".format(trees_dir, i)

    # make tree directory
    try:
        os.mkdir(trees_dir)
    except FileExistsError:
        pass

    # train_data read
    train = Train()
    train.read(train_file)

    # train_data -> inverted_index
    if os.path.exists(index_file):
        print("Already inverted index file exists: {}".format(index_file))
        inverted_index = InvertedIndex(L=train.L)
        inverted_index.read(index_file)

    else:
        inverted_index = train.make_index()
        inverted_index.write(index_file)

    # train_data -> ClassificationTree
    def job(i: int):
        tree = train.make_tree(
            k=settings.NumOfNeighbors,
            TH=settings.ThresholdParameter,
            max_in_leaf=settings.MaxInLeaf,
            inverted_index=inverted_index
        )
        file_name = tree_file_name(i)
        tree.write(file_name)

    if settings.Threads != 1:
        with Pool(processes=settings.Threads) as pool:
            pool.map(job, range(settings.NumOfTrees))
    else:
        for i in range(settings.NumOfTrees):
            job(i)

    # split `data_set` to `sample_list` and `label_list`
    reader = FileReader()
    data_set, N = reader.read(predict_file), reader.N

    sample_list = [data_set[key]['feature'] for key in data_set]
    label_list = [data_set[key]['label'] for key in data_set]

    # predict
    trees = [
        ClassificationTree(
            L=train.L,
            M=train.M,
            k=settings.NumOfNeighbors,
            TH=settings.ThresholdParameter,
            max_in_leaf=settings.MaxInLeaf
        ) for i in range(settings.NumOfTrees)
    ]
    for i in range(settings.NumOfTrees):
        trees[i].read(tree_file_name(i))

    k = settings.KOfPrecision
    predict = Predict()
    predict.load_trees(*trees)
    output_file = input("Output file name: ")

    with open(output_file, "w") as f:
        for sample in sample_list:
            result = predict.predict(sample)
            f.write(','.join([str(index) for index in np.argsort(result)[::-1][:k]])+'\n')

    # validate
    valid = Validate(k, N)
    valid.read(output_file)
    valid.diff(label_list)
