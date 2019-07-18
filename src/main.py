#!/usr/bin/env python3
import sys
import settings
from scripts.train import Train
from scripts.predict import Predict
from scripts.validate import Validate
from scripts.file_reader import FileReader

def main():
    # train
    train = Train()
    train.load(settings.TrainFileName, settings.DEBUG)
    trees = train.make_tree(settings.NumOfTrees, settings.DEBUG)

    # split `test_data_set` to `sample_list` and `label_vector_list`
    reader = FileReader(settings.PredictFileName)
    test_data_set, N_test = reader.read(), reader.N

    sample_list = [data["feature"] for data in test_data_set]
    label_vector_list = [data["label"] for data in test_data_set]

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
