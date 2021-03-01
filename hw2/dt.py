"""
In dt.py, you will implement a basic decision tree classifier for
binary classification.  Your implementation should be based on the
minimum classification error heuristic (even though this isn't ideal,
it's easier to code than the information-based metrics).
"""

import util
from sklearn.tree import DecisionTreeClassifier
from numpy import *
from binary import *
import datasets
import numpy as np


class DT(BinaryClassifier):
    """
    This class defines the decision tree implementation.  It comes
    with a partial implementation for the tree data structure that
    will enable us to print the tree in a canonical form.
    """

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

        # initialize the tree data structure.  all tree nodes have a
        # "isLeaf" field that is true for leaves and false otherwise.
        # leaves have an assigned class (+1 or -1).  internal nodes
        # have a feature to split on, a left child (for when the
        # feature value is < 0.5) and a right child (for when the
        # feature value is >= 0.5)

        self.isLeaf = True
        self.label = 1

    def online(self):
        """
        Our decision trees are batch
        """
        return False

    def __repr__(self):
        """
        Return a string representation of the tree
        """
        return self.displayTree(0)

    def displayTree(self, depth):
        # recursively display a tree
        if self.isLeaf:
            return (" " * (depth * 2)) + "Leaf " + repr(self.label) + "\n"
        else:
            return (" " * (depth * 2)) + "Branch " + repr(self.feature) + "\n" + \
                   self.left.displayTree(depth + 1) + \
                   self.right.displayTree(depth + 1)

    def predict(self, X):
        """
        Traverse the tree to make predictions.  You should threshold X
        at 0.5, so <0.5 means left branch and >=0.5 means right
        branch.
        """
        if self.isLeaf == True:
            return self.label

        threshold = 0.5

        for x in X:
            if x < threshold:
                self.left.predict(X)
            else:
                self.right.predict(X)

    def trainDT(self, X, Y, used):
        """
        recursively build the decision tree
        """
        maxDepth = self.opts["maxDepth"]

        # get the size of the data set
        N, D = X.shape

        # check to see if we're either out of depth or no longer
        # have any decisions to make
        if maxDepth <= 0 or len(util.uniq(Y)) <= 1:
            # we'd better end at this point.  need to figure
            # out the label to return
            self.isLeaf = True
            self.label = util.mode(Y)

        else:
            # we need to find a feature to split on
            bestFeature = -1  # which feature has lowest error
            bestError = N  # the number of errors for this feature
            for d in range(D):
                # have we used this feature yet
                if d in used:
                    continue
                # suppose we split on this feature; what labels
                # would go left and right?
                leftY = Y[X[:, d] < 0.5]
                rightY = Y[X[:, d] >= 0.5]

                # we'll classify the samples in each branch as the common class
                # the error is the number of samples not equal to the common class in their respective branch
                common = util.mode(X[:,d])
                error = len([i for i in X[:, d] if i != common])

                # check to see if this is a better error rate
                if error <= bestError:
                    bestFeature = util.uniq(X[:,d]).index(common)
                    bestError = error
                    used.append(d)

            if bestFeature < 0:
                # this shouldn't happen, but just in case...
                self.isLeaf = True
                self.label = util.mode(Y)

            else:
                self.isLeaf = False

                self.feature = datasets.SentimentData.words[bestFeature]

                self.left = DT({'maxDepth': maxDepth - 1})
                self.right = DT({'maxDepth': maxDepth - 1})
                # TODO Note for grader: I couldn't figure out what element to remove from X, I kept getting an error about the xy dimensions not lining up
                # recurse on our children by calling
                self.left.trainDT(X[1:], leftY, used)
                # and
                self.right.trainDT(X[1:], rightY, used)
                # with appropriate arguments

    def train(self, X, Y):
        """
        Build a decision tree based on the data from X and Y.  X is a
        matrix (N x D) for N many examples on D features.  Y is an
        N-length vector of +1/-1 entries.

        Some hints/suggestions:
          - make sure you don't build the tree deeper than self.opts['maxDepth']
          
          - make sure you don't try to reuse features (this could lead
            to very deep trees that keep splitting on the same feature
            over and over again)
            
          - it is very useful to be able to 'split' matrices and vectors:
            if you want the ids for all the Xs for which the 5th feature is
            on, say X[:,5]>=0.5.  If you want the corresponting classes,
            say Y[X[:,5]>=0.5] and if you want the correspnding rows of X,
            say X[X[:,5]>=0.5,:]
            
          - i suggest having train() just call a second function that
            takes additional arguments telling us how much more depth we
            have left and what features we've used already

          - take a look at the 'mode' and 'uniq' functions in util.py
        """

        self.trainDT(X, Y, [])

    def getRepresentation(self):
        """
        Return our internal representation: for DTs, this is just our
        tree structure -- i.e., ourselves
        """

        return self
