import sys
import numpy as np
from cline import cline

from loss_func import HingeLoss as HingeLoss
from loss_func import Accuracy as Accuracy

class SGD:
    def __init__(self, model, x, y, batch_size=1, lambda_=1.0,
                 learning_rate=0.0001, loss_type='HingeLoss',
                 regularization='L2'):

        self._loss_type = loss_type
        self._regularization = regularization
        self._lambda = lambda_
        self._lrate = learning_rate
        self._bsize = batch_size
        self._model = model
        self.x = x
        self.y = y
        self.used_samples = np.zeros(self.x.shape[0])

    def get_batch(self): # start name of func with _

        idxs = np.random.choice(np.where(self.used_samples == 0)[0], self._bsize)
        batch_x = self.x[idxs]
        batch_y = self.y[idxs]
        self.used_samples[idxs] = 1.0

        return batch_x, batch_y

    def calc_loss(self, batch_x, batch_y, loss=None):

        if not (batch_x.shape[0] == batch_y.shape[0]):
            print 'Input arrays shapes mismatching in \
                    SGD.calc_loss(), line', cline()
            sys.exit()
        if loss is None:
            loss = self._loss_type
        output_loss = 0.0

        # HingeLoss
        if loss == 'HingeLoss':
            batch_y_predicted = self._model.predict(batch_x, binar=False)
            output_loss = np.mean(HingeLoss(batch_y_predicted, batch_y))
            # output_loss /= float(self._bsize)

        # Accuracy
        elif loss == 'Accuracy':
            batch_y_predicted = self._model.predict(batch_x)
            output_loss = Accuracy(batch_y_predicted, batch_y)
        else:
            print 'Unknown loss type in SGD.calc_loss, line', cline()
            sys.exit()



        if self._regularization == None:
            return output_loss
        elif self._regularization == 'L2':
            if not self._loss_type == 'Accuracy':
                output_loss += 0.5 * self._lambda * np.sum(self._model.wb[:-1]**2)
            return output_loss
        else:
            print 'Unknown regularization type in SGD.calc_loss, line', cline()
            sys.exit()

    def grad(self, batch_x, batch_y):
        if not (batch_x.shape[0] == batch_y.shape[0]):
            print 'Input arrays shapes mismatching in \
                    SGD.grad(), line', cline()
            sys.exit()

        grad_matr = np.zeros([self._bsize, self._model.features_size + 1])
        output = np.zeros(self._model.features_size + 1)
        if self._loss_type == 'HingeLoss':
            batch_y_predicted = self._model.predict(batch_x, binar=False)
            current_loss = HingeLoss(batch_y_predicted, batch_y)
            for i in range(self._bsize):
                if current_loss[i] == 0.0:
                    continue
                else:
                    for j in range(self._model.features_size):
                        grad_matr[i, j] = -batch_y[i] * batch_x[i, j]
                    grad_matr[i,self._model.features_size] = - batch_y[i]

            output = np.sum(grad_matr, axis=0)

        if self._regularization == None:
            return output
        elif self._regularization == 'L2':
            return output + self._lambda * self._model.wb

    def step(self, batch_x, batch_y):
        self._model.wb -= self._lrate * self.grad(batch_x, batch_y)

    def make_epoch(self):
        while np.sum(self.used_samples) < self.used_samples.shape[0]:
            batch_x, batch_y = self.get_batch()
            self.step(batch_x, batch_y)

        #back to initial statement
        self.used_samples = np.zeros(self.x.shape[0])
