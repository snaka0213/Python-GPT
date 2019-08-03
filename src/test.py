#!/usr/bin/env python3
import os
import sys
from multiprocessing import Pool

import settings
from scripts.train import Train
from scripts.predict import Predict
from scripts.validate import Validate
from scripts.file_reader import FileReader
from scripts.inverted_index import InvertedIndex
from scripts.learning.classifier import ClassificationTree

def main():
    path, output_file = sys.argv[1], sys.argv[2]
    train_file = "data/" + path + "/train.txt"
    index_file = "data/" + path + "/index.json"
    predict_file = "data/" + path + "/test.txt"
    trees_dir = "data/" + path + "/trees/"

    # train_data load
    train = Train()
    train.load(train_file, settings.DEBUG)

    # split `test_data_set` to `sample_list` and `label_vector_list`
    reader = FileReader(predict_file)
    test_data_set, N_test = reader.read(), reader.N

    sample_list = [test_data_set[key]["feature"] for key in test_data_set]
    label_vector_list = [test_data_set[key]["label"] for key in test_data_set]
    
    # predict
    trees_reloaded = [
        ClassificationTree(
            train.L, train.M,
            settings.NumOfNeighbors,
            settings.MaxInLeaf,
            None
        ) for i in range(settings.NumOfTrees)
    ]
    for i in range(settings.NumOfTrees):
        trees_reloaded[i].open("{}tree_{}.json".format(trees_dir, i))

    predict = Predict()
    predict.load(*trees_reloaded)
    predict.predict(sample_list)
    predict.write(settings.KOfPrecision, output_file)

    # validate
    valid = Validate(settings.KOfPrecision, N_test)
    valid.read(output_file)
    valid.diff(label_vector_list)

if __name__ == "__main__":
    main()
