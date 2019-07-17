#!/usr/bin/env python3
import settings
from scripts.train import Train
from scripts.predict import Predict
from scripts.validate import Validate
from scripts.file_reader import FileReader

def main():
    # train
    train = Train()
    train.load(settings.TrainFileName, settings.DEBUG)
    trees = train.make_tree(settings.DEBUG)

    # split `test_data_set` to `sample_list` and `labels_list`
    reader = FileReader(settings.PredictFileName)
    test_data_set = reader.read()
    N_test = reader.N

    sample_list = [data["feature"] for data in test_data_set]
    labels_list = [data["label"] for data in test_data_set]

    # predict
    predict = Predict()
    predict.load(*trees)
    predict.predict(sample_list)
    predict.write(settings.OutputFileName)

    # validate
    valid = Validate(settings.KOfPrecision, N_test)
    valid.read(settings.OutputFileName)
    valid.diff(labels_list)

if __name__ == "__main__":
    main()
