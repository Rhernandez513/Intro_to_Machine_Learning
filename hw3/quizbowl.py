import sklearn.metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from numpy import *
import datasets

if not datasets.Quizbowl.loaded:
    datasets.loadQuizbowl()

    
print('\n\nRUNNING ON {YOUR CHOSEN} DATA\n')
    
print('training')
# YOUR CODE HERE
print('predicting')
# YOUR CODE HERE

savetxt('predictions.txt', devPred)
