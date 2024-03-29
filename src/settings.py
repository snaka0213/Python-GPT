#!/usr/bin/env python3

### validation ###
# k of `precision@k`
KOfPrecision = 5

### classifier ###
# max number of leafs in `ClassificationTree`
MaxInLeaf = 10

# the number of `ClassificationTree`
NumOfTrees = 50

### KNNG ###
# threshold parameter in Approximate version of `KNNG`
ThresholdParameter = -1

# k in `KNNG`
NumOfNeighbors = 10 # <= MaxInLeaf

### LearnHyperPlane ###
# the number of threads
Threads = 1

# the epsilon in derivative
Epsilon = 1e-8

# the number of epochs in `LearnHyperPlane`
Epoch = 10

# the random sampling size in `LearnHyperPlane`
SampleSize = 10 # <= MaxInLeaf

# the regularization parameter in `LearnHyperPlane`
Lambda = 4

# the initial learning rate in `AdaGrad`
InitialLearningRate = 0.1
