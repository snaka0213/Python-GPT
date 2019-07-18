#!/usr/bin/env python3

# print debug
DEBUG = True

### Data Set ###
# train file
TrainFileName = "data/TestData/train.txt"

# predict file
PredictFileName = "data/TestData/test.txt"

# output file
OutputFileName = "prediction.txt"

### Hyper Parameters ###
# max number of leafs in `ClassificationTree`
MaxInLeaf = 5

# the number of `ClassificationTree`
NumOfTrees = 50

# the number of epochs in `LearnHyperPlane`
Epoch = 10

# the number of random samples in `LearnHyperPlane`
RandomSampleNumber = 10

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# the epsilon in `AdaGrad`
Epsilon = 1e-8

# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.001

# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = 50

# k in `KNNG`
KNNGNumber = 10

# k in `precision@k`
KOfPrecision = 3
