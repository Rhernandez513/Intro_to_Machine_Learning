from math import *
import random
from numpy import *
import matplotlib.pyplot as plt

waitForEnter = False


def generateUniformExample(numDim):
    return [random.random() for d in range(numDim)]


def generateUniformDataset(numDim, numEx):
    return [generateUniformExample(numDim) for n in range(numEx)]


def computeExampleDistance(x1, x2):
    dist = 0.0
    for d in range(len(x1)):
        dist += (x1[d] - x2[d]) * (x1[d] - x2[d])
    return sqrt(dist)


def computeDistances(data):
    N = len(data)
    D = len(data[0])
    dist = []
    for n in range(N):
        for m in range(n):
            dist.append(computeExampleDistance(data[n], data[m]) / sqrt(D))
    return dist
