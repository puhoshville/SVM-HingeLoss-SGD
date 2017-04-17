import sys
import numpy as np
from cline import cline

def HingeLoss(model, X, y_true):
    if not (X.shape[0] == y_true.shape[0]):
        print 'Mismatching of input ndarray input shapes in function "HingeLoss", line:', cline()
        print 'X.shape =', X.shape, 'y_true.shape =', y.shape
        sys.exit()

    y = model.predict(X, binar = False)

    output = 1.0 - (y * y_true)
    output *= np.asarray((output > 0.0),dtype=float)

    return output

def Accuracy(model, X, y_true):
    if not (X.shape[0] == y_true.shape[0]):
        print 'Mismatching of input ndarray input shapes in function "Accuracy", line:', cline()
        print 'X.shape =', X.shape, 'y_true.shape =', y.shape
        sys.exit()

    y = model.predict(X, binar = False)

    output = np.sum(y_true == y) / float(y.shape[0])

    return output
