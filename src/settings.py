#!/usr/bin/env python3

# print debug
DEBUG = True

### Data Set ###
# train file
TrainFileName = "data/Wiki10-31K/wiki10_train.txt"

# predict file
PredictFileName = "data/Wiki10-31K/wiki10_test.txt"

# output file
OutputFileName = "prediction.txt"

### Hyper Parameters ###
# max number of leafs in `ClassificationTree`
MaxInLeaf = 5

# the number of `ClassificationTree`
NumOfTrees = 1

# the number of epochs in `LearnHyperPlane`
EpochNumber = 10

# the number of random samples in `LearnHyperPlane`
RandomSampleNumber = 10

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = 50

# k in `KNNG`
KNNGNumber = 10

# k in `precision@k`
KOfPrecision = 5
