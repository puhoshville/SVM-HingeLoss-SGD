import sys
import numpy as np
from cline import cline
### !!! vmesto model and X lu4she srau y_pred
def HingeLoss(y, y_true):

    if not (y.shape[0] == y_true.shape[0]):
        print 'Mismatching of input ndarray input shapes in function "HingeLoss", line:', cline()
        print 'y.shape =', X.shape, 'y_true.shape =', y.shape
        sys.exit()

    output = 1.0 - (y * y_true)
    output *= np.asarray((output > 0.0),dtype=float)

    return output

def Accuracy(y, y_true):

    if not (y.shape[0] == y_true.shape[0]):
        print 'Mismatching of input ndarray input shapes in function "Accuracy", line:', cline()
        print 'y.shape =', X.shape, 'y_true.shape =', y.shape
        sys.exit()

    output = np.sum(y_true == y) / float(y.shape[0])

    return output
