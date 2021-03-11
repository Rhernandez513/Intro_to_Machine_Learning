from binary import *
from util import *
from numpy import *
import numpy as np

class OVA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for k in range(K):
            self.f.append(mkClassifier())

    def train(self, X, Y):
        for k in range(self.K):
            print("training classifier for {0} versus rest".format(k))
            Yk = 2 * (Y == k) - 1   # +1 if it's k, -1 if it's not k
            self.f[k].fit(X, Yk)  
    
    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for k in range(self.K):
            probs = self.f[k].predict_proba(X.reshape(1, -1))
            if useZeroOne:
                vote[k] += 1 if probs[0,1] > 0.5 else 0
            else:
                vote[k] += probs[0,1]   # weighted vote
        return np.argmax(vote)

    def predictAll(self, X, useZeroOne=False):    
        N,D = X.shape
        Y   = zeros(N, dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
        

class AVA:
    def __init__(self, K, mkClassifier):
        self.f = []
        self.K = K
        for i in range(K):
            self.f.append([])
        for j in range(K):
            for i in range(j):
                self.f[j].append(mkClassifier())

    def train(self, X, Y):
        for i in range(self.K):
            if i == 0:
                continue
            for j in range(i):
                print("training classifier for {0} versus {1}".format(i,j))
                Yij = []
                Xij = []

                for idx, y in enumerate(Y):
                    if y == j:
                        Yij.append(1)
                        Xij.append(X[idx])
                    elif y == i:
                        Yij.append(0)
                        Xij.append(X[idx])

                self.f[i][j].fit(np.array(Xij), np.array(Yij))

    def predict(self, X, useZeroOne=False):
        vote = zeros((self.K,))
        for i in range(self.K):
            for j in range(i):
                p = self.f[i][j].predict_proba(X.reshape(1, -1))
                if useZeroOne:
                    if p[0,1] == 1:
                        vote[i] += 1 if p[0,1] > 0.5 else 0
                    else:
                        vote[j] += 1 if p[0,1] > 0.5 else 0
                else:
                    if p[0,1] == 1:
                        vote[i] += p[0,1]
                    else:
                        vote[j] += 1
        return np.argmax(vote)

    def predictAll(self, X, useZeroOne=False):
        N,D = X.shape
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:], useZeroOne)
        return Y
    
class TreeNode:
    def __init__(self):
        self.isLeaf = True
        self.label  = 0
        self.info   = None

    def setLeafLabel(self, label):
        self.isLeaf = True
        self.label  = label

    def setChildren(self, left, right):
        self.isLeaf = False
        self.left   = left
        self.right  = right
    
    def getLabel(self):
        if self.isLeaf: return self.label
        else: raise Exception("called getLabel on an internal node!")
        
    def getLeft(self):
        if self.isLeaf: raise Exception("called getLeft on a leaf!")
        else: return self.left
        
    def getRight(self):
        if self.isLeaf: raise Exception("called getRight on a leaf!")
        else: return self.right

    def setNodeInfo(self, info):
        self.info = info

    def getNodeInfo(self): return self.info

    def iterAllLabels(self):
        if self.isLeaf:
            yield self.label
        else:
            for l in self.left.iterAllLabels():
                yield l
            for l in self.right.iterAllLabels():
                yield l

    def iterNodes(self):
        yield self
        if not self.isLeaf:
            for n in self.left.iterNodes():
                yield n
            for n in self.right.iterNodes():
                yield n

    def __repr__(self):
        if self.isLeaf:
            return str(self.label)
        l = repr(self.left)
        r = repr(self.right)
        return '[' + l + ' ' + r + ']'
            

def makeBalancedTree(allK):
    if len(allK) == 0:
        raise Exception("makeBalancedTree: cannot make a tree of 0 classes")

    tree = TreeNode()
    
    if len(allK) == 1:
        tree.setLeafLabel(allK[0])
    else:
        split  = len(allK)//2
        leftK  = allK[0:split]
        rightK = allK[split:]
        leftT  = makeBalancedTree(leftK)
        rightT = makeBalancedTree(rightK)
        tree.setChildren(leftT, rightT)

    return tree

class MCTree:
    def __init__(self, tree, mkClassifier):
        self.f = []
        self.tree = tree
        for n in self.tree.iterNodes():
            n.setNodeInfo(mkClassifier())

    def train(self, X, Y):
        for idx, n in enumerate(self.tree.iterNodes()):
            if n.isLeaf:   # don't need to do any training on leaves!
                continue

            # otherwise we're an internal node
            leftLabels  = list(n.getLeft().iterAllLabels())
            rightLabels = list(n.getRight().iterAllLabels())

            print("training classifier for {0} versus {1}".format(leftLabels,rightLabels))
            # compute the training data, store in thisX, thisY
            leftY = [y for y in Y if y in leftLabels]
            rightY = [y for y in Y if y in rightLabels]

            thisY = np.array(leftY + rightY)

            thisX = []
            for idx, x in enumerate(X):
                thisX.append([])
                for dataPoint in x:
                    if dataPoint in leftLabels:
                        thisX[idx].append(1)
                    elif dataPoint in rightLabels:
                        thisX[idx].append(0)
                    else:
                        thisX[idx].append(dataPoint)

            n.getNodeInfo().fit(np.array(thisX), thisY)

    def help_train(self, X, Y, n):
        for idx, n in enumerate(self.tree.iterNodes()):
            if n.isLeaf:   # don't need to do any training on leaves!
               return
        # otherwise we're an internal node
        leftLabels  = list(n.getLeft().iterAllLabels())
        rightLabels = list(n.getRight().iterAllLabels())

        leftY = [y for y in Y if y in leftLabels]
        rightY = [y for y in Y if y in rightLabels]
        n.getNodeInfo().fit(X, Y)
        pass

    def predict(self, X):
        return self.help_predict(X, self.tree)

    def help_predict(self, X, n):
        if n.isLeaf:
            return n.getLabel()
        probs = n.getNodeInfo.predict_proba(X.reshape(1, -1))
        n = n.getLeft() if probs[0, 1] > 0.5 else n.getRight()
        return self.help_predict(X, n)


    def predictAll(self, X):
        N,D = X.shape
        Y   = zeros((N,), dtype=int)
        for n in range(N):
            Y[n] = self.predict(X[n,:])
        return Y

def getMyTreeForWine():
    return makeBalancedTree(20)

