#!/usr/bin/env python3

### Debug ###
DEBUG = True

### validation ###
# k of `precision@k`
KOfPrecision = 5

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
# the number of threads
Threads = 4

# the epsilon in derivative
Epsilon = 1e-8

# the number of epochs in `LearnHyperPlane`
Epoch = 10

# the times of batch learning in `LearnHyperPlane`
BatchTime = 10

# the sample size in `LearnHyperPlane`
SampleSize = 10 # <= MaxInLeaf

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.1

