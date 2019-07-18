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
MaxInLeaf = 10 # >= BatchSize

# the number of `ClassificationTree`
NumOfTrees = 50

# the number of epochs in `LearnHyperPlane`
Epoch = 10

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# the epsilon in `AdaGrad`
Epsilon = 1e-8

# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.1

# the batch size in `AdaGrad`
BatchSize = 10 # <= MaxInLeaf

# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = 50

# k in `KNNG`
KNNGNumber = 10

# k in `precision@k`
KOfPrecision = 3
