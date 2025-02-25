"""
Some useful graphics functions
"""
from numpy import ndarray

import matplotlib.pyplot as plt
import util
import binary

from numpy import *
from pylab import *

def plotLinearClassifier(h, X, Y):
    """
    Draw the current decision boundary, margin and data
    """
    plt.figure(1)
    plt.plot(X[Y>=0.5,0], X[Y>=0.5,1], 'b+',
         X[Y< 0.5,0], X[Y< 0.5,1], 'ro')
    axes = plt.figure(1).get_axes()[0]
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    xmin = xlim[0] + (xlim[1] - xlim[0]) / 100
    xmax = xlim[1] - (xlim[1] - xlim[0]) / 100
    ymin = ylim[0] + (ylim[1] - ylim[0]) / 100
    ymax = ylim[1] - (ylim[1] - ylim[0]) / 100

    if type(h.weights) == ndarray:
        b = 0
        try: b = h.bias
        except AttributeError: pass
        w = h.weights

        #print b
        #print w

        # find the zeros along each axis
        # w0*l + w1*? + b = 0  ==>  ? = -(b + w0*l) / w1
        if w[1] != 0:
            xmin_zero = - (b + w[0] * xmin) / w[1]
            xmax_zero = - (b + w[0] * xmax) / w[1]
        else:
            xmin_zero = 0
            xmax_zero = xmax
        if w[0] != 0:
            ymin_zero = - (b + w[1] * ymin) / w[0]
            ymax_zero = - (b + w[1] * ymax) / w[0]
        else:
            ymin_zero = 0
            ymax_zero = ymax

        # now, two of these should actually be in bounds, figure out which
        inBounds = []
        if ylim[0] <= xmin_zero and xmin_zero <= ylim[1]:
            inBounds.append( (xmin, xmin_zero) )
        if ylim[0] <= xmax_zero and xmax_zero <= ylim[1]:
            inBounds.append( (xmax, xmax_zero) )
        if xlim[0] <= ymin_zero and ymin_zero <= xlim[1]:
            inBounds.append( (ymin_zero, ymin) )
        if xlim[0] <= ymax_zero and ymax_zero <= xlim[1]:
            inBounds.append( (ymax_zero, ymax) )

        #print(inBounds)

        # print(axes)
        if len(inBounds) >= 2:
            plt.plot(X[Y>=0.5,0], X[Y>=0.5,1], 'b+',
                 X[Y< 0.5,0], X[Y< 0.5,1], 'ro',
                 [inBounds[0][0], inBounds[1][0]], [inBounds[0][1], inBounds[1][1]], 'k-')
            #figure(1).set_axes(axes)
            plt.legend(('positive', 'negative', 'hyperplane'))
        else:
            plot(X[Y>=0.5,0], X[Y>=0.5,1], 'b+',
                 X[Y< 0.5,0], X[Y< 0.5,1], 'ro')
            #figure(1).set_axes(axes)
            legend(('positive', 'negative'))



def runOnlineClassifier(h, X, Y):
    N,D = X.shape
    order = range(N)
    util.permute(order)
    plt.plot(X[Y< 0.5,0], X[Y< 0.5,1], 'b+',
         X[Y>=0.5,0], X[Y>=0.5,1], 'ro')
    noStop = False
    for n in order:
        print (Y[n], X[n,:])
        h.nextExample(X[n,:], Y[n])
        plt.hold(True)
        plt.plot([X[n,0]], [X[n,1]], 'ys')
        plt.hold(False)
        if not noStop:
            # v = raw_input()
            v = input()
            if v == "q":
                noStop = True
        plotLinearClassifier(h, X, Y)

