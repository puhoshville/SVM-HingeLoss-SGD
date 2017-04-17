import sys
import numpy as np
from cline import cline

from loss_func import *

class SGD:
    def __init__(self, model, X, y, batch_size = 1, _lambda = 1.0, \
                    learning_rate = 0.0001, loss_type = 'HingeLoss', \
                    regularization = 'L2'):

        self._loss_type = loss_type
        self._regularization = regularization
        self._lambda = _lambda
        self._lrate = learning_rate
        self._bsize = batch_size
        self._model = model

        self.X = X
        self.y = y
        self.used = np.zeros(self.X.shape[0])

        # self.w

    def get_batch(self):
        k = np.random.choice(np.where(self.used == 0)[0], self._bsize)
        batch_X = self.X[k]
        batch_y = self.y[k]
        self.used[k] = 1.0

        return batch_X, batch_y

    def calc_loss(self, batch_X, batch_y, loss = None):
        if not (batch_X.shape[0] == batch_y.shape[0]):
            print 'Input arrays shapes mismatching in \
                    SGD.calc_loss(), line', cline()
            sys.exit()
        if loss is None:
            loss = self._loss_type
        output_loss = 0.0

        # HingeLoss
        if loss == 'HingeLoss':
            output_loss = np.mean(HingeLoss(self._model, batch_X, batch_y))
            #output_loss /= float(self._bsize)

        # Accuracy
        elif loss == 'Accuracy':
            output_loss = Accuracy(self._model, batch_X, batch_y)
        else:
            print 'Unknown loss type in SGD.calc_loss, line', cline()
            sys.exit()



        if self._regularization == None:
            return output_loss
        elif self._regularization == 'L2':
            if not self._loss_type == 'Accuracy':
                output_loss += 0.5 * self._lambda * np.sum(self._model.Wb[:-1]**2)
            return output_loss
        else:
            print 'Unknown regularization type in SGD.calc_loss, line', cline()
            sys.exit()

    def grad(self, batch_X, batch_y):
        if not (batch_X.shape[0] == batch_y.shape[0]):
            print 'Input arrays shapes mismatching in \
                    SGD.grad(), line', cline()
            sys.exit()

        grad_matr = np.zeros([self._bsize, self._model.features_size + 1])

        output = np.zeros(self._model.features_size + 1)

        if self._loss_type == 'HingeLoss':
            tmp = HingeLoss(self._model, batch_X, batch_y)
            for i in range(self._bsize):
                if tmp[i] == 0.0:
                    continue
                else:
                    for j in range(self._model.features_size):
                        grad_matr[i,j] = - batch_y[i] * batch_X[i,j]
                    grad_matr[i,self._model.features_size] = - batch_y[i]
            output = np.sum(grad_matr, axis=0)

        if self._regularization == None:
            return output
        elif self._regularization == 'L2':
            return output + self._lambda * self._model.Wb

    def step(self, batch_X, batch_y):
        self._model.Wb -= self._lrate * self.grad(batch_X, batch_y)

    def make_epoch(self):
        while np.sum(self.used) < self.used.shape[0] - 1:
            batch_X, batch_y = self.get_batch()
            self.step(batch_X, batch_y)

        #back to initial statement
        self.used = np.zeros(self.X.shape[0])
