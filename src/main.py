#!/usr/bin/env python3
import os
import sys

import settings
from scripts.train import Train
from scripts.predict import Predict
from scripts.validate import Validate
from scripts.file_reader import FileReader
from scripts.inverted_index import InvertedIndex

def main():
    # train_data load
    train = Train()
    train.load(settings.TrainFileName, settings.DEBUG)

    # train_data -> inverted_index
    if os.path.exists(settings.InvertedIndexFileName):
        if settings.DEBUG:
            print("Already inverted index file exists: {}".format(settings.InvertedIndexFileName))
        inverted_index = InvertedIndex(TH=settings.ThresholdParameter)
        inverted_index.load(settings.InvertedIndexFileName)

    else:
        inverted_index = train.make_index(TH=settings.ThresholdParameter, debug=settings.DEBUG)
        inverted_index.write(settings.InvertedIndexFileName, debug=settings.DEBUG)
    
    trees = train.make_tree(
        settings.NumOfTrees,
        settings.NumOfNeighbors,
        settings.MaxInLeaf,
        inverted_index,
        settings.DEBUG
    )

    # split `test_data_set` to `sample_list` and `label_vector_list`
    reader = FileReader(settings.PredictFileName)
    test_data_set, N_test = reader.read(), reader.N

    sample_list = [test_data_set[key]["feature"] for key in test_data_set]
    label_vector_list = [test_data_set[key]["label"] for key in test_data_set]

    # predict
    predict = Predict()
    predict.load(*trees)
    predict.predict(sample_list)
    predict.write(settings.KOfPrecision, settings.OutputFileName)

    # validate
    valid = Validate(settings.KOfPrecision, N_test)
    valid.read(settings.OutputFileName)
    valid.diff(label_vector_list)

if __name__ == "__main__":
    sys.setrecursionlimit(4100000)
    main()
