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

# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = 50

# k in `KNNG`
KNNGNumber = 10

# the epsilon in derivative
Epsilon = 1e-8

### AdaGrad ###
# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.1

# the batch size in `AdaGrad`
BatchSizeInAdaGrad = 10 # <= MaxInLeaf

### Adam ###
# some parameters
beta_1 = 0.9
beta_2 = 0.999

# the batch size in `Adam`
BatchSizeInAdam = 100

### Validation ###
# k in `precision@k`
KOfPrecision = 3
