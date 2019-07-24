#!/usr/bin/env python3

### Debug ###
DEBUG = True

### File ###
DataFilePath = "data/Wiki10-31K/"
TrainFileName = DataFilePath + "train.txt"
InvertedIndexFileName = DataFilePath + "index.json"
PredictFileName = DataFilePath + "test.txt"
OutputFileName = "prediction.txt"

### validation ###
# k of `precision@k`
KOfPrecision = 3

### classifier ###
# max number of leafs in `ClassificationTree`
MaxInLeaf = 10 # >= BatchSize

# the number of `ClassificationTree`
NumOfTrees = 50

### knng ###
# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = 50

# k in `KNNG`
NumOfNeighbors = 10

### learn_hyperplane ###
# the epsilon in derivative
Epsilon = 1e-8

# the number of epochs in `LearnHyperPlane`
Epoch = 10 # <= N // BatchSize

# the batch size in `LearnHyperPlane`
BatchSize = 100 # <= N

# the sample size in `LearnHyperPlane`
SampleSize = 10 # <= MaxInLeaf

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.1

